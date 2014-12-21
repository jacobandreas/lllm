package struct;

/**
 * @author jda
 */
public class Pair<A extends Comparable<A>,B extends Comparable<B>> implements Comparable<Pair<A,B>> {

    public final A _1;
    public final B _2;

    public Pair(A _1, B _2) {
        this._1 = _1;
        this._2 = _2;
    }

    int hash = -1;
    public int hashCode() {
        if (hash == -1) {
            hash = _1.hashCode() + 7 * _2.hashCode();
        }
        return hash;
    }

    public boolean equals(Object other) {
        if (!(other instanceof Pair)) return false;
        Pair opair = (Pair)other;
        return opair._1.equals(_1) && opair._2.equals(_2);
    }

    public int compareTo(Pair<A,B> other) {
        int c1 = _1.compareTo(other._1);
        if (c1 != 0) return c1;
        return _2.compareTo(other._2);
    }
}
