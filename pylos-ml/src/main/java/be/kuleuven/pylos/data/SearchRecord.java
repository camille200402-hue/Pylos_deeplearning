package be.kuleuven.pylos.data;

import be.kuleuven.pylos.game.PylosGameState;
import be.kuleuven.pylos.game.PylosPlayerColor;

import java.util.List;

public class SearchRecord {
    public final long position;
    public final List<RootActionVisitData> visitData;

    public SearchRecord(long boardState, PylosPlayerColor currentColor, PylosGameState gameState, List<RootActionVisitData> visitData) {
        this.position = splitBoard(boardState) | playerColorToBits(currentColor) << 60 | gameStateToBits(gameState) << 61;
        this.visitData = visitData;
    }

    /**
     * Splits 60-bit interleaved state into:
     * - Bits [0..29]:  White positions
     * - Bits [30..59]: Black positions
     */
    private static long splitBoard(long state) {
        // Extract even bits directly for white
        long whiteBits = compressEvenBits(state);

        // Shift odd bits down by 1 so they align to even positions, then compress
        long blackBits = compressEvenBits(state >>> 1);

        // Pack them: white in lower 30 bits, black in upper 30 bits
        return whiteBits | (blackBits << 30);
    }

    /**
     * Compresses every even bit (0, 2, 4, ..., 58) into contiguous bits 0 to 29.
     */
    private static long compressEvenBits(long x) {
        x &= 0x5555_5555_5555_5555L;                         // Keep even bits:  .a.b.c.d.e.f.g.h
        x = (x ^ (x >>> 1))  & 0x3333_3333_3333_3333L;        // Pairs:           ..ab..cd..ef..gh
        x = (x ^ (x >>> 2))  & 0x0F0F_0F0F_0F0F_0F0FL;        // Nibbles:         ....abcd....efgh
        x = (x ^ (x >>> 4))  & 0x00FF_00FF_00FF_00FFL;        // Bytes:           ........abcdefgh
        x = (x ^ (x >>> 8))  & 0x0000_FFFF_0000_FFFFL;        // 16-bit blocks
        x = (x ^ (x >>> 16)) & 0x0000_0000_FFFF_FFFFL;        // 32-bit blocks
        return x;
    }

    private static long playerColorToBits(PylosPlayerColor color) {
        return color.ordinal();
    }

    private static long gameStateToBits(PylosGameState gameState) {
        assert gameState == PylosGameState.MOVE || gameState == PylosGameState.REMOVE_FIRST || gameState == PylosGameState.REMOVE_SECOND;
        return gameState.ordinal();
    }
}
