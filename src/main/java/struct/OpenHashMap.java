package struct;

import java.util.Arrays;

/**
 * @author jda
 */
public class OpenHashMap {

    long[] keyArr;
    int[] valueArr;

    int size = 0;

    final static long EMPTY_KEY = 0;

    final static int INITIAL_CAPACITY = 10;
    final static double MAX_LOAD_FACTOR = 0.7;
    final static double RELOAD_FACTOR = 1.35;

    public OpenHashMap(int initialCapacity) {
        keyArr = new long[initialCapacity];
        valueArr = new int[initialCapacity];
    }

    public boolean put(long k, int v) {
//        System.out.println("put " + k + "," + v);
        assert(k != EMPTY_KEY);
        if (1d * size / keyArr.length > MAX_LOAD_FACTOR) {
            rehash();
        }
        return putHelper(k, v, keyArr, valueArr);
    }

    public int get(long k) {
        assert(k != EMPTY_KEY);
        int idx = find(k, keyArr);
        if (keyArr[idx] == EMPTY_KEY) {
            return -1;
        }
        return valueArr[idx];
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return keyArr.length;
    }

    public boolean increment(long k) {
        int idx = find(k, keyArr);
        if (keyArr[idx] == EMPTY_KEY) {
            put(k, 1);
            return true;
        } else {
            valueArr[idx]++;
            return false;
        }
    }

    private void rehash() {
        long[] newKeys = new long[(int)(RELOAD_FACTOR * keyArr.length)];
        System.out.println("rehash from " + keyArr.length + " to " + newKeys.length);
        int[] newValues = new int[newKeys.length];
        Arrays.fill(newKeys, EMPTY_KEY);
        size = 0;
        for (int i = 0; i < keyArr.length; i++) {
            long currKey = keyArr[i];
            if (currKey != EMPTY_KEY) {
                int currVal = valueArr[i];
                putHelper(currKey, currVal, newKeys, newValues);
            }
        }
        keyArr = newKeys;
        valueArr = newValues;
    }

    int find(long k, long[] keys) {
        int idx = (int)(k % keys.length);
        if (idx < 0) idx *= -1;
        long currKey = keys[idx];
        //System.out.println(idx);
//        int i = idx;
//        int t = 1;
        while (currKey != EMPTY_KEY && currKey != k) {
            idx += 13;
            idx %= keys.length;
            currKey = keys[idx];
            //idx++;
            //i = idx + t * t * (1 - 2 * (t % 2));
//            i = idx + t * 13;
//            //System.out.println(idx);
//            i %= keys.length;
//            if (i < 0) i += keys.length;
//            //System.out.println("try " + i);
//            //if (idx == keys.length) idx = 0;
//            currKey = keys[i];
//            t++;
        }
        //System.out.println("ok");
        //System.out.println(t + " probes");
//        return i;
        return idx;
    }

    private boolean putHelper(long k, int v, long[] keys, int[] values) {
        int idx = find(k, keys);
        values[idx] = v;
        long currKey = keys[idx];
        if (currKey == EMPTY_KEY) {
            size++;
            keys[idx] = k;
            return true;
        }
        return false;
    }

}
