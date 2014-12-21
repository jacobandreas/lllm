package struct;

import java.util.ArrayList;
import java.util.List;

/**
 * @author jda
 */
public class SharedKeyArray {

    public final List<RehashListener> listeners;
    public long[] keys;

    private int size;

    private static final long EMPTY_KEY = 0;
    private static final double MAX_LOAD_FACTOR = 0.7;
    private static final double RELOAD_FACTOR = 1.5;

    public SharedKeyArray(int initialCapacity) {
        keys = new long[initialCapacity];
        size = 0;
        listeners = new ArrayList<RehashListener>();
    }

    public void addListener(RehashListener listener) {
        listeners.add(listener);
    }

    public void put(long key) {
    }

}
