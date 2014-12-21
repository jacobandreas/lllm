package struct;

/**
 * @author jda
 */
public class IntSeq implements Comparable<IntSeq> {

    public int[] contents;
    private static final int[] PRIMES = {7, 11, 13, 17, 23};

    public IntSeq(int[] contents) {
        this.contents = contents;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof IntSeq)) return false;
        IntSeq iother = (IntSeq)other;
        if (iother.contents.length != contents.length) return false;
        for (int i = 0; i < contents.length; i++) {
            if (contents[i] != iother.contents[i]) return false;
        }
        return true;
    }

    int hash = -1;
    @Override
    public int hashCode() {
        if (hash == -1) {
            int r = 0;
            for (int i = 0; i < contents.length; i++) {
                r += contents[i] + PRIMES[i % PRIMES.length];
            }
            hash = r;
        }
        return hash;
    }


    @Override
    public int compareTo(IntSeq o) {
        if (contents.length != o.contents.length) {
            return contents.length - o.contents.length;
        }
        for (int i = 0; i < contents.length; i++) {
            if (contents[i] != o.contents[i]) {
                return contents[i] - o.contents[i];
            }
        }
        return 0;
    }

    public String toString() {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < contents.length; i++) {
            b.append(contents[i]);
            if (i < contents.length - 1) {
                b.append("_");
            }
        }
        return b.toString();
    }
}
