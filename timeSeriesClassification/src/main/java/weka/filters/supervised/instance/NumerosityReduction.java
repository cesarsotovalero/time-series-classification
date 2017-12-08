package weka.filters.supervised.instance;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Vector;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.neighboursearch.DTWSearch;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/**
 * <!-- globalinfo-start --> Numerosity reduction, speed up one-nearest neighbor
 * to produce an extremely compact dataset with little or not loss in
 * accuracy.<br/> The original dataset must fit entirely in memory. The
 * attributes to reduce and the percentage of the reduction should be specified.
 * The dataset must have a nominal class attribute.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre> -P &lt;num&gt;
 *  The percentage of the class(es) to reduce (default 10%)</pre>
 *
 * <pre> -D &lt;num&gt;
 *  The Distance Function to use by the filter (default Euclidian Distance)</pre>
 *
 * <pre> -A &lt;num&gt;
 *  The class to reduce (default all)
 *
 *  (default 0)</pre>
 *
 *
 * <!-- options-end -->
 *
 * @author César Soto (csoto@uclv.edu.cu)
 *
 */
public class NumerosityReduction extends Filter implements SupervisedFilter, OptionHandler {

    /**
     * The percentage to sub of the Data Set.
     */
    protected int m_Percent = 10;
    /**
     * The distance function used by the filter.
     */
    protected DistanceFunction m_DistanceFunction = new weka.core.DTWDistance();
    /**
     * Stores which classes to reduce.
     */
    protected Range m_ClassesToReduce = new Range("first-last");
    /**
     * The original instances.
     */
    private LinkedList<InstancesList> m_Instances = new LinkedList<InstancesList>();
    /**
     * The original instances ranked (can have ties).
     */
    private LinkedList<InstancesList> m_InstancesRanked = new LinkedList<InstancesList>();
    /**
     * The original instances re-ranked by priorities (without ties).
     */
    private LinkedList<InstancesList> m_InstancesRankedPriorities = new LinkedList<InstancesList>();

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input instance
     * structure (any instances contained in the object are ignored - only the
     * structure is required).
     * @return true if the outputFormat may be collected immediately
     * @exception Exception if the input format can't be set successfully
     */
    @Override
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        m_ClassesToReduce.setUpper(instanceInfo.numClasses() - 1);
        setOutputFormat(instanceInfo);
        return true;
    }

    /**
     * Compare two instances.
     *
     * @param instance1
     * @param instance2
     * @return
     */
    private boolean compareInstances(Instance instance1, Instance instance2) {
        for (int i = 0; i < instance1.numAttributes(); i++) {
            if (instance1.value(i) != instance2.value(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Remove the duplicate instances of a Data Set.
     *
     * @param data
     * @return
     */
    private Instances removeDuplicateInstances(Instances data) {
        for (int i = 0; i < data.numInstances() - 1; i++) {
            Instance instance = data.get(i);
            int j = i + 1;
            while (j < data.numInstances()) {
                if (compareInstances(instance, data.get(j))) {
                    data.remove(j);
                } else {
                    j++;
                }
            }
        }
        return data;
    }

    /**
     * Give the nearest neighbors for given instance in the Data Set using 1-NN
     * with the chosen distance function.
     *
     * @param instance
     * @param data
     * @param k
     * @return
     * @throws Exception
     */
    private Instances nearestNeighbors(Instance instance, Instances data, int k) throws Exception {
        IBk ibk = new IBk(k);
        NearestNeighbourSearch nearestNeighbourSearch;
        if (m_DistanceFunction instanceof DTWDistance) {
            nearestNeighbourSearch = new DTWSearch(data);
        } else {
            nearestNeighbourSearch = new LinearNNSearch(data);
        }
        nearestNeighbourSearch.setDistanceFunction(m_DistanceFunction);
        ibk.setNearestNeighbourSearchAlgorithm(nearestNeighbourSearch);
        ibk.buildClassifier(data);

        return ibk.getNearestNeighbourSearchAlgorithm().kNearestNeighbours(instance, k);
    }

    /**
     * Return the index of a given Instance
     *
     * @param data
     * @param inst
     * @return
     */
    private int getIndexOf(Instances data, Instance inst) {

        for (int i = 0; i < data.size(); i++) {
            if (compareInstances(data.get(i), inst)) {
                return i;
            }

        }

        return 0;

    }

    /**
     * Initialize the List of instances attribute that record the list of 1-NN
     * nearest neighbors instances calculated.
     *
     * @param data
     * @param k
     * @throws Exception
     */
    private void initializeListOfInstances(Instances data, int k) throws Exception {

        int length = data.size();
        for (int i = 0; i < data.numInstances(); i++) {

            Instance inst = (DenseInstance) data.remove(0);
            Instances nearest = nearestNeighbors(inst, data, k);
            LinkedList<Integer> nearestIndex = new LinkedList<Integer>();
            for (int l = 0; l < nearest.size(); l++) {
                int j = getIndexOf(data, nearest.get(l));
                if (j > length - i - 2) {
                    j = j - (length - i) + 1;
                } else {
                    j = j + i + 1;
                }
                nearestIndex.add(j);

            }
            InstancesList list = new InstancesList((Instance) inst, nearestIndex);

            m_Instances.add(list);

            data.add(inst);
        }

        for (int i = 0; i < m_Instances.size(); i++) {
            LinkedList<Integer> nn = m_Instances.get(i).nearestNeighbors;

            for (int j = 0; j < nn.size(); j++) {

                m_Instances.get(nn.get(j)).setHavingAsItsNN(i);

            }
        }
    }

    /**
     * Assign a rank number to a given instance.
     *
     * @param instance
     * @param data
     * @param k
     * @return
     * @throws Exception
     */
    private void rankInstanceList(InstancesList instance) throws Exception {

        int sum = 0;

        for (int j = 0; j < instance.getHavingAsItsNN().size(); j++) {
            LinkedList<Integer> Hnn = instance.getHavingAsItsNN();
            if (instance.getInstance().classValue() == m_Instances.get(Hnn.get(j)).getInstance().classValue()) {
                sum++;
            } else {
                sum += -2;
            }
        }

        instance.setRank(sum);

        insertInstanceList(instance, m_InstancesRanked);
    }

    /**
     * Get a list of instances with the same rank value in ranked list from a
     * given index.
     *
     * @param index The index when begin the search in the list.
     * @return A list of instances with the same rank value in given ranked
     * list.
     */
    private LinkedList<InstancesList> getTies(int index) {

        LinkedList<InstancesList> list = new LinkedList<InstancesList>();
        list.add(m_InstancesRanked.get(index));

        double rank = m_InstancesRanked.get(index).rank;
        int i = index + 1;
        while (i < m_InstancesRanked.size() && m_InstancesRanked.get(i).rank == rank) {
            list.add(m_InstancesRanked.get(i));
            i++;
        }

        return list;
    }

    /**
     * Re-Rank a list of instances with the same rank value.
     *
     * @param data All the data instances.
     * @param ties A list of instances with the same rank value.
     * @return A list of instances re-ranked.
     */
    private LinkedList<InstancesList> rankTies(Instances data, LinkedList<InstancesList> ties) {

        LinkedList<InstancesList> resultList = new LinkedList<InstancesList>();
        m_DistanceFunction.setInstances(data);

        for (int i = 0; i < ties.size(); i++) {

            double newRank = 0;
            InstancesList inst = ties.get(i);
            inst.setRank(0);

            for (int j = 0; j < inst.getHavingAsItsNN().size(); j++) {

                double d = m_DistanceFunction.distance(inst.instance, data.get(inst.getHavingAsItsNN().get(j)));
                newRank += 1 / (Math.pow(d, 2));
            }
            inst.setRank(newRank);
            insertInstanceList(inst, resultList);
        }

        return resultList;
    }

    /**
     * If there are instances with the same rank, this method break the tie by
     * assigning different priorities to them.
     *
     * @param data
     */
    private void reRankInstanceList(Instances data) {

        for (int i = 0; i < m_InstancesRanked.size();) {

            LinkedList<InstancesList> ties = getTies(i);

            if (ties.size() > 1) {
                ties = rankTies(data, ties);
            }

            m_InstancesRankedPriorities.addAll(ties);
            i += ties.size();

        }
    }// end method

    /**
     * This is the main algorithm, Rank Reduction works in two steps: ranking
     * and thresholding. We first assign ranks to all instances based on their
     * contribution for the classification of the training set. In the
     * rankInstanceList step, we begin by removing duplicated instances, if any.
     * Then we apply one-nearest-neighbor classification on the training set. We
     * assign lowest ranks to misclassified instances. If two instances have the
     * same rank, we break the tie by assigning different priorities to them
     * using the reRankInstanceList method.
     *
     * @param data
     * @param k
     * @throws Exception
     */
    private void rankBasedReduction(Instances data, int k) throws Exception {

        // remove any duplicate instances from data
        removeDuplicateInstances(data);

        // leave-one-out 1-NN classification on data
        initializeListOfInstances(data, k);

        // ranking all instances in the data
        for (int j = 0; j < data.numInstances(); j++) {
            Instance inst = data.remove(0);
            rankInstanceList(m_Instances.get(j));
            data.add(inst);
        }

        // re-ranking the instances (removing ties)
        reRankInstanceList(data);
    }

    /**
     * Find the index in a List for an ordered insertion in it.
     *
     * @param valor
     * @param inicio
     * @param fin
     * @param lista
     * @return
     */
    private int busqBinPos(double valor, int inicio, int fin, LinkedList<InstancesList> lista) {
        if (lista.size() == 0) {
            return 0;
        }
        if (inicio == fin) {
            if (lista.get(fin).getRank() >= valor) {
                return fin;
            } else {
                return fin + 1;
            }
        }
        int medio = (inicio + fin) / 2;
        if (lista.get(medio).getRank() >= valor) {
            return busqBinPos(valor, inicio, medio, lista);
        } else {
            return busqBinPos(valor, medio + 1, fin, lista);
        }
    }

    /**
     * Insert in a list (of InstancesList) orderly.
     *
     * @param n
     * @param lista
     */
    private void insertInstanceList(InstancesList n, LinkedList<InstancesList> lista) {
        int i = busqBinPos(n.getRank(), 0, lista.size() - 1, lista);
        lista.add(i, n);
    }

    /**
     * Signify that this batch of input to the filter is finished. Output() may
     * now be called to retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @exception IllegalStateException if no input structure has been defined
     */
    @Override
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!isFirstBatchDone()) {

            rankBasedReduction(getInputFormat(), 1);

            int inRange = 0;
            for (int i = 0; i < m_InstancesRankedPriorities.size(); i++) {
                Instance inst = m_InstancesRanked.get(i).instance;

                if (m_ClassesToReduce.isInRange((int) inst.value(inst.classIndex()))) {
                    inRange++;
                }
            }

            int rest = inRange - (inRange * m_Percent / 100);
            boolean out = true;

            for (int i = m_InstancesRankedPriorities.size() - 1; i >= 0; i--) {
                Instance inst = m_InstancesRanked.get(i).instance;

                if (out && m_ClassesToReduce.isInRange((int) inst.value(inst.classIndex()))) {
                    push(m_InstancesRanked.get(i).instance);
                    rest--;
                } else if (!m_ClassesToReduce.isInRange((int) inst.value(inst.classIndex()))) {
                    push(m_InstancesRanked.get(i).instance);
                }

                if (rest == 0) {
                    out = false;
                    rest = -1;
                }
            }

            m_FirstBatchDone = true;
            flushInput();
        }
        m_NewBatch = true;
        return (numPendingOutput() != 0);

    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration listOptions() {
        Vector result = new Vector();

        result.addElement(new Option(
                "\tSpecify the percentage of instance to remove (default 10)",
                "P", 10, "-P <num>"));

        // List the Distance Function
        if ((m_DistanceFunction != null) && (m_DistanceFunction instanceof OptionHandler)) {
            Enumeration enu = ((OptionHandler) m_DistanceFunction).listOptions();

            result.addElement(new Option("", "", 0, "\nOptions specific the "
                    + "Distance Function " + m_DistanceFunction.getClass().getName() + ":"));
            while (enu.hasMoreElements()) {
                result.addElement((Option) enu.nextElement());
            }
        }
        return result.elements();
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String tmpStr;

        tmpStr = Utils.getOption('P', options);
        if (tmpStr.length() != 0) {
            setPercentageToRemove(Integer.parseInt(tmpStr));
        } else {
            setPercentageToRemove(10);
        }

        String distanceString = Utils.getOption('D', options);
        if (distanceString.length() == 0) {
            distanceString = weka.core.DTWDistance.class.getName();
        }
        String[] distanceStringSpec = Utils.splitOptions(distanceString);
        if (distanceStringSpec.length == 0) {
            throw new Exception("Invalid Distance Function specification string");
        }
        m_DistanceFunction.setOptions(distanceStringSpec);
        setDistanceFunction(m_DistanceFunction);

    }

    /**
     * Gets the current settings for the attribute selection (instances to
     * reduce, distance function) etc.
     *
     * @return
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();
        String[] distanceOptions = new String[0];

        result.add("-P");
        result.add("" + getPercentageToRemove());

        // get Distance Function
        if (m_DistanceFunction instanceof OptionHandler) {
            distanceOptions = ((OptionHandler) m_DistanceFunction).getOptions();
        }

        result.add("-D");
        result.add(getDistanceFunction().getClass().getName()
                + " " + Utils.joinOptions(distanceOptions));

        return result.toArray(new String[result.size()]);
    }

    /**
     * Sets the percentage to remove of the Data Set.
     *
     * @param
     */
    public void setPercentageToRemove(int newPercentage) {
        m_Percent = newPercentage;
    }

    /**
     * Gets the the percentage to remove of the Data Set.
     *
     * @return
     */
    public int getPercentageToRemove() {
        return m_Percent;
    }

    /**
     * Get the name of the Distance Function
     *
     * @return the name of the Distance Function as a string
     */
    public DistanceFunction getDistanceFunction() {

        return m_DistanceFunction;
    }

    /**
     * Set Distance Function class
     *
     * @param search the search class to use
     */
    public void setDistanceFunction(DistanceFunction distance) {
        m_DistanceFunction = distance;
    }

    /**
     * Gets the current range selection
     *
     * @return a string containing a comma separated list of ranges
     */
    public String getClassesIndicesToReduce() {

        return m_ClassesToReduce.getRanges();
    }

    /**
     * Sets which attributes are to be Discretized (only numeric attributes
     * among the selection will be Discretized).
     *
     * @param rangeList a string representing the list of attributes. Since the
     * string will typically come from a user, attributes are indexed from 1.
     * <br> eg: first-3,5,6-last
     * @exception IllegalArgumentException if an invalid range list is supplied
     */
    public void setClassesIndicesToReduce(String rangeList) {

        m_ClassesToReduce.setRanges(rangeList);
    }

    /**
     * Returns a string describing this filter.
     *
     * @return a description of the filter suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "Numerosity reduction, to speed up one-nearestneighbor algorithms classification "
                + "to produce an extremely compact dataset with little or no loss in accuracy.\n"
                + "The original dataset must fit entirely in memory."
                + " The attributes to reduce and its percentage may be specified. "
                + " The dataset must have a nominal class attribute. Useful for "
                + "removing outliers.\n";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String percentageToRemoveTipText() {
        return "The percentage of the class(es) to reduce (default 10%)";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String distanceFunctionTipText() {
        return "The Distance Function (by default: DTWDistance) to use by the filter. In the case of DTWDistance,"
                + " DTWSearch is used as NearestNeighbourAlgoritm to speed up the filter. "
                + "For EuclidianDistance, LinnearNNSearch is used as NearestNeighbourAlgoritm. ";
    }

    /**
     * Internal class used to record the instances ranking.
     *
     */
    class InstancesList {

        private Instance instance;
        private LinkedList<Integer> nearestNeighbors;
        private LinkedList<Integer> havingAsItsNN;
        private double rank;

        InstancesList(Instance instance, LinkedList<Integer> nearestNeighbors) {
            this.instance = instance;
            this.rank = 0;
            this.nearestNeighbors = nearestNeighbors;
            this.havingAsItsNN = new LinkedList<Integer>();
        }

        /**
         * Return the instance.
         *
         * @return the instance
         */
        public Instance getInstance() {
            return instance;
        }

        /**
         * Return the rank.
         *
         * @return the rank
         */
        public double getRank() {
            return rank;
        }

        /**
         * Set the rank.
         *
         * @param rank the rank to set
         */
        public void setRank(double rank) {
            this.rank = rank;
        }

        /**
         * Return the list with having as its nearest neighbor
         *
         * @return the havingAsItsNN
         */
        public LinkedList<Integer> getHavingAsItsNN() {
            return havingAsItsNN;
        }

        /**
         * Set the index of the Instances that having as is nearest neighbor
         *
         * @param i
         */
        public void setHavingAsItsNN(int i) {
            this.havingAsItsNN.add(i);
        }
    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        runFilter(new NumerosityReduction(), argv);

    }
}
