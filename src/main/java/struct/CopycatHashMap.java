package struct;

import java.util.Arrays;

/**
 * @author jda
 */
public class CopycatHashMap {

    private final OpenHashMap base;
    private final int initCapacity;

    int size;

    final int[] valueArr;

    static final int EMPTY_VAL = -1;

    public CopycatHashMap(OpenHashMap base) {
        this.base = base;
        this.initCapacity = base.keyArr.length;
        valueArr = new int[initCapacity];
        Arrays.fill(valueArr, EMPTY_VAL);
        size = 0;
    }

    public boolean put(long k, int v) {
//        System.out.println("put " + k + "," + v);
        assert(v != EMPTY_VAL);
        assert(k != OpenHashMap.EMPTY_KEY);
        assert(1d * initCapacity / valueArr.length > OpenHashMap.MAX_LOAD_FACTOR);
        assert(base.keyArr.length == initCapacity);
        return putHelper(k, v);
    }

    public int get(long k) {
        assert(k != OpenHashMap.EMPTY_KEY);
        int idx = base.find(k, base.keyArr);
        if (valueArr[idx] == EMPTY_VAL) {
            return -1;
        }
        return valueArr[idx];
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return valueArr.length;
    }

    public boolean increment(long k) {
        int idx = base.find(k, base.keyArr);
        if (valueArr[idx] == EMPTY_VAL) {
            put(k, 1);
            return true;
        } else {
            valueArr[idx]++;
            return false;
        }
    }

    private boolean putHelper(long k, int v) {
        //int idx = find(k, keys);
        int idx = base.find(k, base.keyArr);
        long currVal = valueArr[idx];
        valueArr[idx] = v;
        if (currVal == EMPTY_VAL) {
            size++;
            return true;
        }
        return false;
    }

}
