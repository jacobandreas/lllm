package struct;

/**
 * @author jda
 */
public interface RehashListener {
    public void onRehash(long[] oldKeys, long[] newKeys);
}
