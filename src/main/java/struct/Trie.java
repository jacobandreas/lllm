package struct;

/**
 * @author jda
 */
public class Trie {

    public static final int BRANCHES = 255;

    public Trie[] children = null;
    public double[] values = null;

    public void ensureChild(int index) {
        if (children == null) children = new Trie[BRANCHES];
        if (children[index] == null) children[index] = new Trie();
        //System.out.println("after ensuring, me: " + this + " children: " + children);
    }

    public void ensureValues() {
        if (values == null) values = new double[BRANCHES];
    }

}
