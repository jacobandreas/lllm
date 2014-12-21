package struct;

import java.util.Map;
import java.util.TreeMap;

/**
 * @author jda
 */
public class Index {

    private Map<String,Integer> entries;
    private boolean locked;

    public Index() {
        entries = new TreeMap<String,Integer>();
        locked = false;
    }

    public int add(String s) {
        assert(!locked);
        if (entries.containsKey(s)) {
            return entries.get(s);
        }
        int val = entries.size();
        entries.put(s, val);
        return val;
    }

    public int get(String s) {
        return entries.get(s);
    }

    public void lock() {
        locked = true;
    }

    public int size() {
        return entries.size();
    }
}

// divide training up into chunks
// build sorted ngram & continuation lists for each chunk
// merge sort on disk
// pack

