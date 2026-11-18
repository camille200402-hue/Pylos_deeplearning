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

    private static int mapActionToActionId(Action action) {
        // TODO: Map every possible action to a dense index
        return 0;
    }

    public static final class ValueNetwork implements ValueFunction {
        public static final int INPUT_SIZE = 60;
        public static final int HIDDEN_SIZE = 32;
        public static final int OUTPUT_SIZE = 1;

        public static final int QA = 255;
        public static final int QB = 64;

        public static final Path WEIGHTS_PATH = Path.of("value/quantised.bin");
//        public static final NetworkWeights WEIGHTS = NetworkWeights.loadAtClassInit(WEIGHTS_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        @Override
        public double evaluate(PylosBoard board, PylosPlayerColor perspective) {
            // TODO: implement value network inference
            return 0;
        }
    }

    public static final class PolicyNetwork implements PolicyFunction {
        public static final int INPUT_SIZE = 60;
        public static final int HIDDEN_SIZE = 32;
        public static final int OUTPUT_SIZE = -1; // TODO: set to output size of policy network

        public static final int QA = 255;
        public static final int QB = 64;

        public static final Path WEIGHTS_PATH = Path.of("policy/quantised.bin");
//        public static final NetworkWeights WEIGHTS = NetworkWeights.loadAtClassInit(WEIGHTS_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        @Override
        public double[] priors(PylosBoard board, PylosPlayerColor toMove, List<Action> actions) {
            // TODO: implement policy network inference
            return new double[0];
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
