package struct;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * @author jda
 */
public class Continuation {

    //private SortedMap<IntSeq,SortedMap<Integer,Integer>> contents;
    private Map<Pair<IntSeq,Integer>,Integer> counts;
    private SortedSet<Pair<IntSeq,Integer>> keys;

    public Continuation() {
        //contents = new TreeMap<IntSeq,SortedMap<Integer,Integer>>();
        counts = new HashMap<Pair<IntSeq,Integer>,Integer>();
        keys = new TreeSet<Pair<IntSeq,Integer>>();
    }

    public void put(IntSeq history, int next) {
//        if (!contents.containsKey(history)) {
//            contents.put(history, new TreeMap<Integer,Integer>());
//        }
//        if (!contents.get(history).containsKey(next)) {
//            contents.get(history).put(next, 0);
//        }
//        contents.get(history).put(next, contents.get(history).get(next) + 1);
        Pair<IntSeq,Integer> p = new Pair<IntSeq,Integer>(history, next);
        Integer count = counts.get(p);
        if (count == null) {
            count = 0;
            counts.put(p, 0);
            keys.add(p);
        }
        counts.put(p, count + 1);
    }

    public void write(String path) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(path));
//            for (Map.Entry<IntSeq, SortedMap<Integer, Integer>> entry : contents.entrySet()) {
//                for (Map.Entry<Integer, Integer> prefix : entry.getValue().entrySet()) {
//                    writer.write(entry.getKey().toString() + "," + prefix.getKey().toString() + "," + prefix.getValue().toString());
//                    writer.write("\n");
//                }
//            }
            for (Pair<IntSeq,Integer> key : keys) {
                writer.write(key._1.toString());
                writer.write(",");
                writer.write(key._2);
                writer.write(",");
                writer.write(counts.get(key));
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
