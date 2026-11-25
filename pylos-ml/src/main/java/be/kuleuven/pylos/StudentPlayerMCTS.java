package be.kuleuven.pylos;

import be.kuleuven.pylos.data.RootActionVisitData;
import be.kuleuven.pylos.data.SearchRecord;
import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class StudentPlayerMCTS extends PylosPlayer {

    // Evaluates the current board.
    // Returns the win value in [-1,1] for the given perspective color:
    // -1 is a loss, 0 is neutral, and 1 is a win.
    public interface ValueFunction {
        double evaluate(PylosBoard board, PylosPlayerColor perspective);
    }

    // Returns a prior (policy value) for every action, in the same order as the given list.
    public interface PolicyFunction {
        double[] priors(PylosBoard board, PylosPlayerColor toMove, List<Action> actions);
    }

    // Baseline value: difference in reserve spheres, mapped to [-1,1].
    public static class ReserveDifferenceValue implements ValueFunction {
        @Override
        public double evaluate(PylosBoard board, PylosPlayerColor perspective) {
            int difference = board.getReservesSize(perspective) - board.getReservesSize(perspective.other());
            return difference / (double) board.SPHERES_PER_PLAYER;
        }
    }

    // Baseline policy: every action gets the same prior.
    public static class UniformPolicy implements PolicyFunction {
        @Override
        public double[] priors(PylosBoard board, PylosPlayerColor toMove, List<Action> actions) {
            double[] priors = new double[actions.size()];
            Arrays.fill(priors, 1.0 / actions.size());
            return priors;
        }
    }

    private static final double FPU_VALUE = 0.0;

    private final ValueFunction valueFunction;
    private final PolicyFunction policyFunction;
    private final double explorationWeight;
    private final TimeManager timeManager;

    public PylosGameSimulator simulator;

    // Training data: one record per search, outcome filled in afterwards via setOutcome
    private boolean recordSearches = false;
    private final List<SearchRecord> searchRecords = new ArrayList<>();

    public StudentPlayerMCTS(ValueFunction valueFunction, PolicyFunction policyFunction, double explorationWeight, TimeManager timeManager) {
        this.valueFunction = valueFunction;
        this.policyFunction = policyFunction;
        this.explorationWeight = explorationWeight;
        this.timeManager = timeManager;
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        simulator = new PylosGameSimulator(game.getState(), PLAYER_COLOR, board);
        search(board).execute(game);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        simulator = new PylosGameSimulator(game.getState(), PLAYER_COLOR, board);
        search(board).execute(game);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        simulator = new PylosGameSimulator(game.getState(), PLAYER_COLOR, board);
        search(board).execute(game);
    }

    // MCTS
    public Action search(PylosBoard board) {
        Node root = new Node(null, null, 1.0);
        expand(root, board);

        long deadline = System.nanoTime() + timeManager.getTimeLimit().toNanos();
        int playouts = 0;

        while (true) {
            // 1. Selection
            Node leaf = select(root);
            // 2. Expansion + 3. Evaluation
            double value;
            if (simulator.getState() == PylosGameState.COMPLETED) {
                value = terminalValue(PLAYER_COLOR);
            } else {
                expand(leaf, board);
                value = valueFunction.evaluate(board, PLAYER_COLOR);
            }
            // 4. Backpropagation
            backPropagate(leaf, value, PLAYER_COLOR);

            if (System.nanoTime() - deadline >= 0) {
                break;
            }

            if (playouts++ >= timeManager.getPlayouts()) {
                break;
            }
        }

        if (recordSearches) {
            searchRecords.add(new SearchRecord(board.toLong(), simulator.getColor(), simulator.getState(), createRootVisitData(root)));
        }
        return root.getMostVisitedChild().action;
    }

    // ============
    // 1. Selection
    // ============

    // Descend the tree along the highest PUCT values, simulating each action on the way.
    private Node select(Node root) {
        Node node = root;
        while (!node.children.isEmpty()) {
            node = findChildWithHighestPUCT(node);
            node.action.simulate(simulator);
        }
        return node;
    }

    // PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child)
    private Node findChildWithHighestPUCT(Node node) {
        double sqrtParentVisits = Math.sqrt(Math.max(1, node.visitCount));
        Node best = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (Node child : node.children) {
            double q = child.visitCount == 0 ? FPU_VALUE : child.valueSum / child.visitCount;
            double u = explorationWeight * child.prior * sqrtParentVisits / (1 + child.visitCount);
            if (q + u > bestValue) {
                bestValue = q + u;
                best = child;
            }
        }
        return best;
    }

    // ============
    // 2. Expansion
    // ============

    // Create a child for every available action, with its prior from the policy function.
    private void expand(Node node, PylosBoard board) {
        PylosPlayerColor toMove = simulator.getColor();
        List<Action> actions = generateAllActions(board, toMove, new ArrayList<>());
        if (actions.isEmpty()) {
            return;
        }
        double[] priors = policyFunction.priors(board, toMove, actions);
        node.children = new ArrayList<>(actions.size());
        for (int i = 0; i < actions.size(); i++) {
            node.children.add(new Node(actions.get(i), node, priors[i]));
        }
    }

    public List<Action> generateAllActions(PylosBoard board, PylosPlayerColor color, List<Action> actions) {
        PylosGameState gameState = simulator.getState();
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        switch (gameState) {
            case MOVE -> {
                // Moving spheres already on the board to a higher level
                for (PylosSphere sphere : spheres) {
                    if (!sphere.isReserve() && sphere.canMove()) {
                        for (PylosLocation location : locations) {
                            if (sphere.canMoveTo(location)) {
                                actions.add(new Action(ActionType.UPGRADE, sphere, sphere.getLocation(),
                                        location, gameState, color));
                            }
                        }
                    }
                }
                // Adding a sphere from the reserve
                PylosSphere reserveSphere = board.getReserve(color);
                for (PylosLocation location : locations) {
                    if (reserveSphere.canMoveTo(location)) {
                        actions.add(new Action(ActionType.ADD, reserveSphere, null, location,
                                gameState, color));
                    }
                }
            }
            case REMOVE_FIRST, REMOVE_SECOND -> {
                ActionType type = gameState == PylosGameState.REMOVE_FIRST
                        ? ActionType.REMOVE_FIRST
                        : ActionType.REMOVE_SECOND;
                for (PylosSphere sphere : spheres) {
                    if (sphere.canRemove()) {
                        actions.add(new Action(type, sphere, sphere.getLocation(), null, gameState, color));
                    }
                }
                // Passing is only allowed for the second removal
                if (gameState == PylosGameState.REMOVE_SECOND) {
                    actions.add(new Action(ActionType.PASS, null, null, null, gameState, color));
                }
            }
        }
        return actions;
    }

    // Value of a finished game for the given perspective
    private double terminalValue(PylosPlayerColor perspective) {
        return simulator.getWinner() == perspective ? 1.0 : -1.0;
    }

    // ==================
    // 4. Backpropagation
    // ==================

    // value is the win value in [-1,1] for `perspective`; each node stores it from the
    // perspective of the player who made the action leading to that node.
    private void backPropagate(Node leaf, double value, PylosPlayerColor perspective) {
        Node current = leaf;
        while (current.action != null) {
            current.visitCount++;
            current.valueSum += current.action.color == perspective ? value : -value;
            current.action.undo(simulator);
            current = current.parent;
        }
        current.visitCount++;
    }
    // Accumulate sparse Layer 0 weights for all active features in our 162-feature encoding
    //TODO only 1 of 2 weights on
    private static void accumulateFeatures(PylosBoard board, PylosPlayerColor perspective, short[] acc, short[] l0Weights, int hiddenSize) {
        PylosLocation[] locations = board.getLocations();

        // 1. Occupancy (0..59) & Blocked Spheres (60..119)
        for (PylosLocation loc : locations) {
            PylosSphere sphere = loc.getSphere();
            if (sphere != null) {
                int idx = locationIndex(loc);
                boolean isOwn = (sphere.PLAYER_COLOR == perspective);

                // Occupancy feature
                addWeightRow(acc, l0Weights, (isOwn ? 0 : 30) + idx, hiddenSize);

                // Blocked feature (sphere cannot move/be removed because another sphere rests on it)
                if (!sphere.canMove()) {
                    addWeightRow(acc, l0Weights, (isOwn ? 60 : 90) + idx, hiddenSize);
                }
            }
        }

        // 2. 2x2 Square Threats (120..147) & Playable Upper Cells (148..161)
        // Iterate over all 14 upper cells (z = 1..3), each sits directly above a 2x2 square on level z-1
        int upperIdx = 0; // 0..13
        for (int z = 1; z <= 3; z++) {
            int upperWidth = 4 - z;
            for (int y = 0; y < upperWidth; y++) {
                for (int x = 0; x < upperWidth; x++) {
                    // Check the 4 supporting cells on level z - 1
                    int ownCount = 0;
                    int oppCount = 0;
                    for (int dy = 0; dy <= 1; dy++) {
                        for (int dx = 0; dx <= 1; dx++) {
                            PylosSphere s = board.getBoardLocation(x + dx, y + dy, z - 1).getSphere();
                            if (s != null) {
                                if (s.PLAYER_COLOR == perspective) ownCount++;
                                else oppCount++;
                            }
                        }
                    }

                    // 3-of-4 threat in the 2x2 square on level z - 1
                    if (ownCount == 3 && oppCount == 0) {
                        addWeightRow(acc, l0Weights, 120 + upperIdx, hiddenSize);
                    } else if (oppCount == 3 && ownCount == 0) {
                        addWeightRow(acc, l0Weights, 134 + upperIdx, hiddenSize);
                    }

                    // Upper cell itself is playable (empty + 4 supports present)
                    PylosLocation upperLoc = board.getBoardLocation(x, y, z);
                    if (upperLoc.getSphere() == null && (ownCount + oppCount == 4)) {
                        addWeightRow(acc, l0Weights, 148 + upperIdx, hiddenSize);
                    }

                    upperIdx++;
                }
            }
        }
    }

    private static void addWeightRow(short[] acc, short[] l0Weights, int featureIdx, int hiddenSize) {
        int rowOffset = featureIdx * hiddenSize;
        for (int h = 0; h < hiddenSize; h++) {
            acc[h] = (short) (acc[h] + l0Weights[rowOffset + h]);
        }
    }
    public void setRecordSearches(boolean recordSearches) {
        this.recordSearches = recordSearches;
    }

    public List<SearchRecord> getSearchRecords() {
        return searchRecords;
    }

    public enum ActionType {
        REMOVE_FIRST,
        REMOVE_SECOND,
        PASS,
        ADD,
        UPGRADE,
    }

    public static class Action {
        public final ActionType type;
        public final PylosSphere sphere;
        public final PylosLocation fromLocation, toLocation;
        public final PylosGameState previousState;
        public final PylosPlayerColor color;

        Action(ActionType type, PylosSphere sphere, PylosLocation fromLocation,
               PylosLocation toLocation, PylosGameState previousState, PylosPlayerColor color) {
            this.type = type;
            this.sphere = sphere;
            this.fromLocation = fromLocation;
            this.toLocation = toLocation;
            this.previousState = previousState;
            this.color = color;
        }

        void simulate(PylosGameSimulator simulator) {
            switch (type) {
                case ADD, UPGRADE -> simulator.moveSphere(sphere, toLocation);
                case REMOVE_FIRST, REMOVE_SECOND -> simulator.removeSphere(sphere);
                case PASS -> simulator.pass();
            }
        }

        void undo(PylosGameSimulator simulator) {
            switch (type) {
                case ADD -> simulator.undoAddSphere(sphere, previousState, color);
                case UPGRADE -> simulator.undoMoveSphere(sphere, fromLocation, previousState, color);
                case REMOVE_FIRST -> simulator.undoRemoveFirstSphere(sphere, fromLocation, previousState, color);
                case REMOVE_SECOND -> simulator.undoRemoveSecondSphere(sphere, fromLocation, previousState, color);
                case PASS -> simulator.undoPass(previousState, color);
            }
        }

        void execute(PylosGameIF game) {
            switch (type) {
                case ADD, UPGRADE -> game.moveSphere(sphere, toLocation);
                case REMOVE_FIRST, REMOVE_SECOND -> game.removeSphere(sphere);
                case PASS -> game.pass();
            }
        }
    }

    public static class Node {
        public final Action action;
        public final Node parent;
        public final double prior;
        public List<Node> children = Collections.emptyList();
        public int visitCount = 0;
        public double valueSum = 0;

        Node(Action action, Node parent, double prior) {
            this.action = action;
            this.parent = parent;
            this.prior = prior;
        }

        Node getMostVisitedChild() {
            Node best = null;
            for (Node child : children) {
                if (best == null || child.visitCount > best.visitCount) {
                    best = child;
                }
            }
            return best;
        }
    }

    public static class TimeManager {
        int playouts;
        Duration timeLimit;

        public TimeManager(int playouts) {
            this.playouts = playouts;
            this.timeLimit = Duration.ofDays(1);
        }

        public TimeManager(Duration timeLimit) {
            this.timeLimit = timeLimit;
            this.playouts = Integer.MAX_VALUE;
        }

        public Duration getTimeLimit() {
            return timeLimit;
        }

        public int getPlayouts() {
            return playouts;
        }
    }

    private static List<RootActionVisitData> createRootVisitData(Node rootNode) {
        List<RootActionVisitData> visitData = new ArrayList<>();
        for (Node child : rootNode.children) {
            int actionId = mapActionToActionId(child.action);
            visitData.add(new RootActionVisitData(actionId, child.visitCount));
        }
        return visitData;
    }

    private static int locationIndex(PylosLocation loc) {
        // Level offsets: z=0 -> 0 (16 cells), z=1 -> 16 (9 cells), z=2 -> 25 (4 cells), z=3 -> 29 (1 cell)
        int levelOffset = switch (loc.Z) {
            case 0 -> 0;
            case 1 -> 16;
            case 2 -> 25;
            case 3 -> 29;
            default -> throw new IllegalArgumentException("Invalid Z: " + loc.Z);
        };
        int width = 4 - loc.Z;
        return levelOffset + loc.Y * width + loc.X;
    }

    /**
     * Action ID mapping (total = 991):
     *   0          : PASS
     *   1 .. 30    : ADD (1 + toIdx)
     *   31 .. 60   : REMOVE_FIRST (31 + fromIdx)
     *   61 .. 90   : REMOVE_SECOND (61 + fromIdx)
     *   91 .. 990  : UPGRADE (91 + fromIdx * 30 + toIdx)
     */
    private static int mapActionToActionId(Action action) {
        return switch (action.type) {
            case PASS -> 0;
            case ADD -> 1 + locationIndex(action.toLocation);
            case REMOVE_FIRST -> 31 + locationIndex(action.fromLocation);
            case REMOVE_SECOND -> 61 + locationIndex(action.fromLocation);
            case UPGRADE -> 91 + locationIndex(action.fromLocation) * 30 + locationIndex(action.toLocation);
        };
    }

    public static final class ValueNetwork implements ValueFunction {
        public static final int INPUT_SIZE = 162;
        public static final int HIDDEN_SIZE = 32;
        public static final int OUTPUT_SIZE = 1;

        public static final int QA = 255;
        public static final int QB = 64;

        public static final Path WEIGHTS_PATH = Path.of("pylos-ml/src/main/training/checkpoints/value-162-32-50/quantised.bin");
        public static final NetworkWeights WEIGHTS = NetworkWeights.loadAtClassInit(WEIGHTS_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        @Override
        public double evaluate(PylosBoard board, PylosPlayerColor perspective) {
            // done
            // make array representing first layer with predecided weight
            short[] acc = Arrays.copyOf(WEIGHTS.l0Bias, HIDDEN_SIZE);

            //multiply by value where 1
            accumulateFeatures(board, perspective, acc, WEIGHTS.l0Weights, HIDDEN_SIZE);

            int sum = WEIGHTS.l1Bias[0]; // add baseline bias
            // for all nodes cap the value with min and max and add weighted value to sum
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                short clamped = acc[h];
                if (clamped < 0) {

                    clamped = 0;
                } else if (clamped > QA) {
                    clamped = QA;
                }
                sum += clamped * WEIGHTS.l1Weights[h];
            }

            // make scale from -1 to 1
            return Math.tanh(sum / (double) (QA * QB));
        }
    }

    public static final class PolicyNetwork implements PolicyFunction {
        public static final int INPUT_SIZE = 120;
        public static final int HIDDEN_SIZE = 32;
        public static final int OUTPUT_SIZE = 991;

        public static final int QA = 255;
        public static final int QB = 64;

        public static final Path WEIGHTS_PATH = Path.of("pylos-ml/src/main/training/checkpoints/policy-162-32-50/quantised.bin");
        public static final NetworkWeights WEIGHTS = NetworkWeights.loadAtClassInit(WEIGHTS_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        @Override
        public double[] priors(PylosBoard board, PylosPlayerColor toMove, List<Action> actions) {
            int numActions = actions.size();
            if (numActions == 0) {
                return new double[0];
            }

            // initilaze array with nodesweights
            short[] acc = Arrays.copyOf(WEIGHTS.l0Bias, HIDDEN_SIZE);

            // look at weights other then 0
            accumulateFeatures(board, toMove, acc, WEIGHTS.l0Weights, HIDDEN_SIZE);

            // cap value
            short[] activated = new short[HIDDEN_SIZE];
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                short val = acc[h];
                if (val < 0) {
                    activated[h] = 0;
                } else if (val > QA) {
                    activated[h] = QA;
                } else {
                    activated[h] = val;
                }
            }


            double[] logits = new double[numActions]; // store score of each move
            double maxLogit = Double.NEGATIVE_INFINITY; // track highest score until then
            double scale = (double) (QA * QB); //scaling

            // loop each legal action
            for (int a = 0; a < numActions; a++) {
                // get action
                int actionId = mapActionToActionId(actions.get(a));
                // jump to right index
                int weightOffset = actionId * HIDDEN_SIZE;
                // add starting bias
                int sum = WEIGHTS.l1Bias[actionId];
                // add to score
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    sum += activated[h] * WEIGHTS.l1Weights[weightOffset + h];
                }
                // scale it and get best move
                double logit = sum / scale;
                logits[a] = logit;
                if (logit > maxLogit) {
                    maxLogit = logit;
                }
            }

            //makes scores into probabilities
            double[] priors = new double[numActions];
            double expSum = 0.0;
            for (int a = 0; a < numActions; a++) {
                double expVal = Math.exp(logits[a] - maxLogit); // no negatives and higher numbers stand out and - max to keep between 0 and 1
                priors[a] = expVal;
                expSum += expVal;
            }
            // normalize
            for (int a = 0; a < numActions; a++) {
                priors[a] /= expSum;
            }

            return priors;
        }
    }

    public static final class NetworkWeights {
        public final short[] l0Weights;
        public final short[] l0Bias;
        public final short[] l1Weights;
        public final short[] l1Bias;

        private NetworkWeights(short[][] parameters) {
            l0Weights = parameters[0];
            l0Bias = parameters[1];
            l1Weights = parameters[2];
            l1Bias = parameters[3];
        }

        private static NetworkWeights loadAtClassInit(Path path, int inputSize, int hiddenSize, int outputSize) {
            try {
                return load(path, inputSize, hiddenSize, outputSize);
            } catch (IOException e) {
                throw new UncheckedIOException("Could not load network weights from "
                        + path.toAbsolutePath(), e);
            }
        }

        // save.rs exports headerless little-endian i16 parameters, padded with zeros to 64 bytes.
        public static NetworkWeights load(Path path, int inputSize, int hiddenSize, int outputSize) throws IOException {
            long payloadBytes = Short.BYTES * ((long) inputSize * hiddenSize + hiddenSize
                    + (long) hiddenSize * outputSize + outputSize);
            long expectedBytes = (payloadBytes + 63) / 64 * 64;
            byte[] bytes = Files.readAllBytes(path);
            if (bytes.length != expectedBytes) {
                throw new IOException("Weight file " + path + " does not match dimensions "
                        + inputSize + "x" + hiddenSize + "x" + outputSize
                        + ": expected " + expectedBytes + " bytes including 64-byte alignment, got "
                        + bytes.length + " bytes");
            }

            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            short[][] parameters = {
                    new short[inputSize * hiddenSize], new short[hiddenSize],
                    new short[outputSize * hiddenSize], new short[outputSize]
            };
            for (short[] parameter : parameters) {
                for (int i = 0; i < parameter.length; i++) {
                    parameter[i] = buffer.getShort();
                }
            }
            while (buffer.hasRemaining()) {
                if (buffer.get() != 0) {
                    throw new IOException("Weight file " + path + " has nonzero padding at byte "
                            + (buffer.position() - 1));
                }
            }
            return new NetworkWeights(parameters);
        }
    }
}
