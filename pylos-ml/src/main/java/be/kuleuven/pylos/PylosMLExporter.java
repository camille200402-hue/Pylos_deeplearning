package be.kuleuven.pylos;

import be.kuleuven.pylos.data.PlayedDataGenGame;
import be.kuleuven.pylos.data.RootActionVisitData;
import be.kuleuven.pylos.data.SearchRecord;
import be.kuleuven.pylos.game.PylosPlayerColor;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * One PYLOSDAT writer per collection thread. Not shared between threads.
 *
 * <p>The only header is the ASCII magic PYLOSDAT. All integers are big-endian.
 * A game is (whiteResult: u8, positionCount: u16), followed by positions:
 * (packedPosition: u64, entryCount: u16, [actionId: u16, visitWeight: u16]*).
 * White results are loss=0, draw=1, win=2. Nonempty policies have maximum 65535.
 * Visit weights are ceil(rawVisits * 65535 / maximumRawVisits).
 * SearchRecord.position already contains the agreed occupancy/turn/phase bits.
 *
 * <p>Writes to path.pdat.partial, and publishes path.pdat on successful close.
 * A failed write leaves the partial file for inspection. Existing output files
 * are never replaced. Only one game is buffered while validating/encoding it.
 */
public final class PylosMLExporter implements AutoCloseable {
    public static final int MAX_U16 = 0xffff;
    private static final long BOARD_MASK = (1L << 30) - 1;
    private static final byte[] MAGIC = "PYLOSDAT".getBytes(StandardCharsets.US_ASCII);

    private final Path outputPath;
    private final Path partialPath;
    private final DataOutputStream output;
    private boolean failed;
    private boolean closed;

    public PylosMLExporter(Path outputPath) throws IOException {
        this.outputPath = Objects.requireNonNull(outputPath).toAbsolutePath().normalize();
        if (!this.outputPath.getFileName().toString().endsWith(".pdat")) {
            throw new IllegalArgumentException("output filename must end in .pdat");
        }
        partialPath = this.outputPath.resolveSibling(this.outputPath.getFileName() + ".partial");
        Files.createDirectories(this.outputPath.getParent());
        if (Files.exists(this.outputPath)) {
            throw new FileAlreadyExistsException(this.outputPath.toString());
        }
        DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(partialPath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)));
        try {
            stream.write(MAGIC);
        } catch (IOException error) {
            try {
                stream.close();
            } catch (IOException closeError) {
                error.addSuppressed(closeError);
            }
            throw error;
        }
        output = stream;
    }

    /** Append a completed game; null winner denotes a draw. */
    public void writeGame(PlayedDataGenGame game) throws IOException {
        if (closed || failed) throw new IllegalStateException("exporter is closed or failed");
        try {
            Objects.requireNonNull(game, "game");
            List<SearchRecord> positions = Objects.requireNonNull(game.records, "positions");
            checkU16(positions.size(), "position count");
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (DataOutputStream record = new DataOutputStream(bytes)) {
                record.writeByte(game.winner == null ? 1 : game.winner == PylosPlayerColor.LIGHT ? 2 : 0);
                record.writeShort(positions.size());
                for (SearchRecord position : positions) {
                    Objects.requireNonNull(position, "position");
                    validatePosition(position.position);
                    List<RootActionVisitData> visits = position.visitData == null ? List.of() : position.visitData;
                    int[] weights = quantizeVisits(visits);
                    record.writeLong(position.position);
                    record.writeShort(visits.size());
                    for (int i = 0; i < visits.size(); i++) {
                        record.writeShort(visits.get(i).actionId());
                        record.writeShort(weights[i]);
                    }
                }
            }
            bytes.writeTo(output);
            output.flush();
        } catch (IOException | RuntimeException error) {
            failed = true;
            throw error;
        }
    }

    private static void checkU16(int value, String name) {
        if (value < 0 || value > MAX_U16) {
            throw new IllegalArgumentException(name + " must fit an unsigned 16-bit integer");
        }
    }

    private static void validatePosition(long position) {
        if (position < 0 || ((position >>> 61) & 3) == 3) {
            throw new IllegalArgumentException("invalid packed phase or reserved position bit");
        }
        long white = position & BOARD_MASK;
        long black = (position >>> 30) & BOARD_MASK;
        if ((white & black) != 0) {
            throw new IllegalArgumentException("white and black occupancy overlap");
        }
    }

    private static int[] quantizeVisits(List<RootActionVisitData> visits) {
        checkU16(visits.size(), "policy entry count");
        int[] weights = new int[visits.size()];
        if (visits.isEmpty()) return weights;

        Set<Integer> ids = new HashSet<>();
        int maximum = 0;
        for (RootActionVisitData entry : visits) {
            Objects.requireNonNull(entry, "policy entry");
            checkU16(entry.actionId(), "action ID");
//            if (!ids.add(entry.actionId())) throw new IllegalArgumentException("duplicate action ID");
            if (entry.visitCount() < 0) throw new IllegalArgumentException("negative visit count");
            maximum = Math.max(maximum, entry.visitCount());
        }
        if (maximum == 0) throw new IllegalArgumentException("nonempty policy must have positive visits");

        // Use long before multiplication. The product plus maximum - 1 fits
        // long for every nonnegative int count; division rounds up exactly.
        for (int i = 0; i < visits.size(); i++) {
            long scaled = visits.get(i).visitCount() * (long) MAX_U16;
            weights[i] = (int) Math.ceilDiv(scaled, maximum);
        }
        return weights;
    }

    /** Close once, then rename the complete stream without replacing any file. */
    @Override
    public void close() throws IOException {
        if (closed) return;
        closed = true;
        try {
            output.close();
        } catch (IOException error) {
            failed = true;
            throw error;
        }
        if (!failed) Files.move(partialPath, outputPath);
    }
}
