package edu.berkeley.nlp.assignments.assign1.student;

import java.io.*;
import java.util.*;

import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import struct.Continuation;
import struct.Index;
import struct.IntSeq;
import struct.Trie;

public class OldLmFactory implements LanguageModelFactory
{

    public static final int ORDER = 3;
    public static final String START_SYMBOL = "<s>";
    public static final int BATCH_SIZE = 100000;
	public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {

        Index vocabIndex = buildVocabIndex(trainingData);
        int nBatches = writeBatchCounts(trainingData, vocabIndex);
        mergeBatchCounts(nBatches);
        buildTrie();

        return null; // TODO Construct an exact LM implementation here.
    }

    private void mergeBatchCounts(int nBatches) {
        int rounds = (int)Math.ceil(Math.log(nBatches) / Math.log(2));
        for (int round = 0; round < rounds; round++) {
            int roundBatches = (int)Math.ceil(nBatches * Math.pow(2, -round));
            for (int order = 1; order <= ORDER; order++) {
                for (int batch = 0; batch < roundBatches; batch += 2) {
                    if (batch+1 < roundBatches) {
                        String f1 = sortName(order, batch, round);
                        String f2 = sortName(order, batch+1, round);
                        String outf = sortName(order, batch / 2, round + 1);
                        //System.out.println("Merge " + batch + " " + (batch+1) + " -> " + batch / 2);
                        mergeFiles(f1, f2, outf);
                    } else {
                        String f1 = sortName(order, batch, round);
                        String outf = sortName(order, batch / 2, round+1);
                        copyFile(f1, outf);
                    }
                }
            }
        }
    }

    private void copyFile(String inf, String outf) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(inf));
            BufferedWriter writer = new BufferedWriter(new FileWriter(outf));
            String l;
            while ((l = reader.readLine()) != null) {
                writer.write(l);
                writer.write("\n");
            }
            reader.close();
            writer.close();
        } catch(IOException e) {
            e.printStackTrace();
        }

    }

    private void mergeFiles(String f1, String f2, String outf) {
        try {
            BufferedReader reader1 = new BufferedReader(new FileReader(f1));
            BufferedReader reader2 = new BufferedReader(new FileReader(f2));
            BufferedWriter writer = new BufferedWriter(new FileWriter(outf));

            String l1 = reader1.readLine();
            String l2 = reader2.readLine();

            while (l1 != null && l2 != null) {
                String[] l1parts = l1.split(",");
                String[] l2parts = l2.split(",");
                String[] l1prefixParts = l1parts[0].split("_");
                String[] l2prefixParts = l2parts[0].split("_");
                if (l1prefixParts[0].equals("")) {
                    l1prefixParts = new String[0];
                    l2prefixParts = new String[0];
                }
                int[] l1ids = new int[l1prefixParts.length + 1];
                int[] l2ids = new int[l2prefixParts.length + 1];
                for (int i = 0; i < l1prefixParts.length; i++) {
                    l1ids[i] = Integer.parseInt(l1prefixParts[i]);
                    l2ids[i] = Integer.parseInt(l2prefixParts[i]);
                }
                l1ids[l1ids.length - 1] = Integer.parseInt(l1parts[1]);
//                System.out.println(f2);
//                System.out.println(Arrays.toString(l2parts));
                l2ids[l2ids.length - 1] = Integer.parseInt(l2parts[1]);
                IntSeq s1 = new IntSeq(l1ids);
                IntSeq s2 = new IntSeq(l2ids);
                int comp = s1.compareTo(s2);
                if (comp < 0) {
                    writer.write(l1);
                    l1 = reader1.readLine();
                } else if (comp > 0) {
                    writer.write(l2);
                    l2 = reader2.readLine();
                } else {
                    writer.write(l1parts[0] + "," + l1parts[1] + "," + (Integer.parseInt(l1parts[2]) + Integer.parseInt(l2parts[2])));
                    l1 = reader1.readLine();
                    l2 = reader2.readLine();
                }
                writer.write("\n");
            }

            if (l1 != null) {
                while (l1 != null) {
                    writer.write(l1);
                    writer.write("\n");
                    l1 = reader1.readLine();
                }
            }
            if (l2 != null) {
                while (l2 != null) {
                    writer.write(l2);
                    writer.write("\n");
                    l2 = reader1.readLine();
                }
            }

            reader1.close();
            reader2.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String sortName(int order, int batch, int round) {
        //return "work/order" + order + "_batch" + batch + "_round" + round;
        return "work/round" + round + "_order" + order + "_batch" + batch;
    }

    private void buildTrie() {

    }

    private int writeBatchCounts(Iterable<List<String>> trainingData, Index vocabIndex) {

        int batch = 0;
        Iterator<List<String>> trainingIter = trainingData.iterator();
        while (trainingIter.hasNext()) {
            System.out.println("Starting batch " + batch);
            Continuation[] continuations = new Continuation[ORDER];
            for (int i = 0; i < continuations.length; i++) {
                continuations[i] = new Continuation();
            }
            for (int l = 0; l < BATCH_SIZE && trainingIter.hasNext(); l++) {
                List<String> line = trainingIter.next();
                for (int w = 0; w < line.size(); w++) {
                    int[] wordIds = new int[ORDER];
                    for (int o = 0; o < ORDER; o++) {
                        if (w - o - 1 < 0) {
                            wordIds[o] = vocabIndex.get(START_SYMBOL);
                        } else {
                            wordIds[o] = vocabIndex.get(line.get(w - o));
                        }
                    }

                    for (int order = 1; order <= ORDER; order++) {
                        IntSeq suffix = new IntSeq(Arrays.copyOfRange(wordIds,0,order-1));
                        int prefix = wordIds[order-1];
                        continuations[order-1].put(suffix,prefix);
                    }
                }
            }

            for (int i = 0; i < continuations.length; i++) {
                continuations[i].write(sortName(i+1, batch, 0));
            }
            batch++;
        }
        return batch;
	}

    public static Index buildVocabIndex(Iterable<List<String>> trainingData) {
        Index index = new Index();
        for (List<String> sent : trainingData) {
            for (String word : sent) {
                index.add(word);
            }
        }
        index.add(START_SYMBOL);
        return index;
    }

}
