package be.kuleuven.pylos.data;

import be.kuleuven.pylos.game.PylosPlayerColor;

import java.util.List;

public class PlayedDataGenGame {
    public PylosPlayerColor winner;
    public List<SearchRecord> records;

    public PlayedDataGenGame(PylosPlayerColor winner, List<SearchRecord> records) {
        this.winner = winner;
        this.records = records;
    }

    public int positionCount() {
        return records.size();
    }
}


