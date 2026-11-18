package be.kuleuven.pylos;

import be.kuleuven.pylos.data.PlayedDataGenGame;
import be.kuleuven.pylos.data.SearchRecord;
import be.kuleuven.pylos.game.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


public class PylosMLCollect {
    public static final String EXPORT_PATH = "pylos-ml/src/main/training/data";
    public static final int RANDOM_OPENING_ACTIONS = 6;
    private static final int PROGRESS_INTERVAL_SECONDS = 1;

    static final int GAMES_PER_THREAD = 1000;
    static final int THREADS = Math.max(Runtime.getRuntime().availableProcessors() - 2, 2);

    public static void main(String[] args) {
        play();
    }

    static void play() {
        Path root = Path.of(EXPORT_PATH);
        String name = "run_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"));
        Path directory = root.resolve(name);
        try {
            Files.createDirectories(root);
            for (int suffix = 1; ; suffix++) {
                try {
                    Files.createDirectory(directory);
                    break;
                } catch (FileAlreadyExistsException e) {
                    directory = root.resolve(name + "_" + suffix);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Could not create the collection output directory", e);
        }
        Path outputDirectory = directory;

        int workerCount = THREADS;
        int totalGames = Math.multiplyExact(GAMES_PER_THREAD, workerCount);
        AtomicInteger gamesPlayed = new AtomicInteger();
        AtomicLong positionsGenerated = new AtomicLong();
        AtomicBoolean stopRequested = new AtomicBoolean();
        CountDownLatch collectionFinished = new CountDownLatch(1);
        long startTime = System.nanoTime();

        Thread shutdownHook = new Thread(() -> {
            stopRequested.set(true);
            System.out.println("Stop requested; finishing active games...");
            boolean interrupted = false;
            while (true) {
                try {
                    collectionFinished.await();
                    break;
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }, "pylos-ml-collection-shutdown");
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        try {
            System.out.printf("Collecting %d games (%d threads, %d games per thread, %d random opening actions)%n",
                    totalGames, workerCount, GAMES_PER_THREAD, RANDOM_OPENING_ACTIONS);
            System.out.println("Output: " + outputDirectory.toAbsolutePath());
            System.out.println("Press Ctrl+C to finish active games and stop.");

            try (ExecutorService pool = Executors.newFixedThreadPool(workerCount)) {
                CompletionService<Void> completion = new ExecutorCompletionService<>(pool);
                int finishedWorkers = 0;
                try {
                    for (int worker = 0; worker < workerCount; worker++) {
                        int threadId = worker;
                        completion.submit(() -> {
                            Random random = new Random();
                            Path outputFile = outputDirectory.resolve(threadId + ".pdat");
                            try (PylosMLExporter exporter = new PylosMLExporter(outputFile)) {
                                for (int gameIndex = 0; gameIndex < GAMES_PER_THREAD && !stopRequested.get(); gameIndex++) {
                                    PlayedDataGenGame playedGame = playGame(random);
                                    exporter.writeGame(playedGame);
                                    positionsGenerated.addAndGet(playedGame.positionCount());
                                    gamesPlayed.incrementAndGet();
                                }
                            }
                            return null;
                        });
                    }
                    pool.shutdown();

                    long lastProgress = startTime;
                    while (finishedWorkers < workerCount) {
                        Future<Void> finished = completion.poll(PROGRESS_INTERVAL_SECONDS, TimeUnit.SECONDS);
                        if (finished != null) {
                            finished.get(); // Propagate worker failures, including output errors.
                            finishedWorkers++;
                        }
                        long now = System.nanoTime();
                        if (now - lastProgress >= TimeUnit.SECONDS.toNanos(PROGRESS_INTERVAL_SECONDS)) {
                            printProgress(gamesPlayed.get(), totalGames, positionsGenerated.get(), now - startTime,
                                    stopRequested.get() ? "finishing active games" : null);
                            lastProgress = now;
                        }
                    }
                } finally {
                    if (finishedWorkers < workerCount) pool.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Game collection interrupted", e);
            } catch (ExecutionException e) {
                throw new IllegalStateException("Game collection failed", e.getCause());
            }

            boolean stoppedEarly = gamesPlayed.get() < totalGames;
            printProgress(gamesPlayed.get(), totalGames, positionsGenerated.get(), System.nanoTime() - startTime,
                    stoppedEarly ? "stopped" : null);
            System.out.println((stoppedEarly ? "Stopped early" : "Collection complete") + ": saved "
                    + gamesPlayed.get() + " games to " + outputDirectory.toAbsolutePath());
        } finally {
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException e) {
                // JVM shutdown is already in progress; release the waiting hook.
            } finally {
                collectionFinished.countDown();
            }
        }
    }

    private static PlayedDataGenGame playGame(Random random) {
        StudentPlayerMCTS light = new StudentPlayerMCTS(new StudentPlayerMCTS.ReserveDifferenceValue(), new StudentPlayerMCTS.UniformPolicy(), 1.0, new StudentPlayerMCTS.TimeManager(2000));
        StudentPlayerMCTS dark = new StudentPlayerMCTS(new StudentPlayerMCTS.ReserveDifferenceValue(), new StudentPlayerMCTS.UniformPolicy(), 1.0, new StudentPlayerMCTS.TimeManager(2000));
        light.setRecordSearches(true);
        dark.setRecordSearches(true);
        PylosBoard board = new PylosBoard();
        PylosGame game = new PylosGame(board, light, dark, random);
        for (int i = 0; i < RANDOM_OPENING_ACTIONS && game.getState() == PylosGameState.MOVE; i++) {
            List<PylosLocation> usable = new ArrayList<>();
            for (PylosLocation location : board.getLocations()) {
                if (location.isUsable()) usable.add(location);
            }
            if (usable.isEmpty()) break;
            PylosPlayerColor color = i % 2 == 0 ? PylosPlayerColor.LIGHT : PylosPlayerColor.DARK;
            game.moveSphere(board.getReserve(color), usable.get(random.nextInt(usable.size())));
        }

        PylosPlayerColor winningColor;
        try {
            game.play();
            winningColor = game.getWinner() == null ? null : game.getWinner().PLAYER_COLOR;
        } catch (PylosGameCrashedException e) {
            // A crashing player forfeits, just as in Battle and SPRT.
            winningColor = e.getOpponentPlayer().PLAYER_COLOR;
        }

        List<SearchRecord> records = new ArrayList<>();
        records.addAll(light.getSearchRecords());
        records.addAll(dark.getSearchRecords());
        return new PlayedDataGenGame(winningColor, records);
    }

    private static void printProgress(int gamesPlayed, int totalGames, long positionsGenerated, long elapsedNanos,
                                      String remaining) {
        double elapsedSeconds = elapsedNanos / 1_000_000_000.0;
        double positionsPerSecond = elapsedSeconds > 0 ? positionsGenerated / elapsedSeconds : 0;
        if (remaining == null) {
            remaining = "unknown";
            if (gamesPlayed > 0) {
                long seconds = (long) Math.ceil(elapsedSeconds * (totalGames - gamesPlayed) / gamesPlayed);
                remaining = String.format(Locale.ROOT, "%02d:%02d:%02d", seconds / 3600, seconds / 60 % 60, seconds % 60);
            }
        }
        System.out.printf(Locale.ROOT, "Games: %d/%d | Positions: %d | Positions/s: %.1f | Remaining: %s%n",
                gamesPlayed, totalGames, positionsGenerated, positionsPerSecond, remaining);
    }
}
