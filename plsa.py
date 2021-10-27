import collections
import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0
        
    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        
        with open(self.documents_path, 'r') as file:
            for line in file:
                self.documents.append(line.split())
            file.close()
        self.number_of_documents =np.shape(self.documents)[0]

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        str_num = ['0', '1', '2']
        flatten_list = np.char.lower(sum(self.documents, []))
        for i in range(np.shape(flatten_list)[0]):
            if flatten_list[i] not in self.vocabulary and flatten_list[i] not in str_num:
                self.vocabulary.append(flatten_list[i])
        self.vocabulary_size = np.shape(self.vocabulary)[0]
        return self.vocabulary #['mount', 'rainier', 'seattle', 'willis', 'tower', 'chicago'] # 是不是要capital seattle & chicago

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        self.term_doc_matrix = []
        for d in range(len(self.documents)):

            tempA = []
            for v in range(len(self.vocabulary)):
                tempA.append(self.documents[d].count(self.vocabulary[v]))
            self.term_doc_matrix.append(tempA)
        
        #print(self.term_doc_matrix)
        return self.term_doc_matrix

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        #z_matri = np.reshape(np.random.randint(0, number_of_topics-1, len(self.documents)),(-1, 1)) random Z
        
        document_topic_prob = np.random.uniform(0, 1, (self.number_of_documents, number_of_topics)) #P(z | d) is Πd,j
        self.document_topic_prob = document_topic_prob / np.reshape(document_topic_prob.sum(axis = 1), (-1,1)) 
        #print(self.document_topic_prob)

        topic_word_prob = np.random.uniform(0, 1, (number_of_topics, len(self.vocabulary))) #P(w | z) = P(w|Θ)
        self.topic_word_prob = topic_word_prob / np.reshape(topic_word_prob.sum(axis = 1), (-1,1))
   
        #print(self.topic_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob)


    #def initialize(self, number_of_topics, random=False):
    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        #print("Initializing...")

        if random:
            #print('random')
            self.initialize_randomly(number_of_topics)
            
        else:
            #print('else')
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self, number_of_topics):
        """ The E-step updates P(z | w, d)
        """
        #print("E step stars.....")
        
        # ############################
        # your code here
        # ############################
        self.theta_word_pi_pro = np.ones((self.vocabulary_size, self.number_of_documents, number_of_topics)) #[vocabulary_size, number_of_documents, number_of_topics]
        self.e_step_result = np.ones((self.vocabulary_size, self.number_of_documents, number_of_topics)) #P(z | w, d)
    
        for i in range(self.vocabulary_size):
            self.theta_word_pi_pro[i] = self.document_topic_prob * self.topic_word_prob[:, i]
            self.e_step_result[i] = self.theta_word_pi_pro[i] / np.reshape(np.sum(self.theta_word_pi_pro[i], axis= 1),(-1,1))

        #print('e_step_result:','\n',  self.e_step_result)
        #print('Expectation step done!')
        return self.e_step_result
        
    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        #print("M step stars.....")

        # ############################
        # your code here
        # ############################
        
        all_sum = 0
        product_wordocc_pi_colle = np.ones((self.number_of_documents, self.vocabulary_size, number_of_topics))
    
        for i in range(self.number_of_documents):
            #updating pi.....
            #updating pi.....
            #updating pi.....
            
            product_wordoccurence_pi = self.e_step_result[:,i,:] * np.reshape(self.term_doc_matrix[i], (-1, 1))
            product_wordoccurence_pi_sum = np.sum(product_wordoccurence_pi, axis=0)
            new_pi = product_wordoccurence_pi_sum / np.sum(product_wordoccurence_pi_sum)
            self.document_topic_prob[i] = new_pi #update pi

            product_wordocc_pi_colle[i] = product_wordoccurence_pi
            all_sum += np.sum(product_wordoccurence_pi, axis=0) #[j, j']
        #print(product_wordocc_pi_colle)
        
        #print('updated pi done !!!')
        # update P(w | z) #self.topic_word_prob
        # ############################
        # your code here
        # ############################
        #updating p(n+1)(w|θ).....
        #updating p(n+1)(w|θ).....
        #updating p(n+1)(w|θ).....
        collection_sum = np.sum(product_wordocc_pi_colle, axis=0)
        self.topic_word_prob = (collection_sum / np.sum(collection_sum, axis=0)).T
        print("new opic_word_prob" ,self.topic_word_prob)
        #print('updated P(w|Θ) !!!')
        #print('maximization step done !!!')

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################

        sum_theta_word_pi = np.sum(self.theta_word_pi_pro, axis=2) #(topics, document_numbers)
        log_like = np.log(sum_theta_word_pi).T * self.term_doc_matrix
        self.likelihoods.append(np.sum(log_like))
        #return self.likelihoods
        #print(np.log(sum_theta_word_pi))
        

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ('#'*20 + "EM iteration begins..." + '#'*20)
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        #self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float) # ori
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=float)
        # P(z | d) P(w | z)
        
        self.initialize(number_of_topics, random=True)
        print ('#'*20 + "Initialize Done !!!" + '#'*20)
        # Run the EM algorithm
        current_likelihood = 0.0
        
        for iteration in range(max_iter):
            Corpus.expectation_step(self, number_of_topics)
            Corpus.maximization_step(self, number_of_topics)
            Corpus.calculate_likelihood(self, number_of_topics)

            print("Iteration #" + str(iteration + 1) + ': '+ str(self.likelihoods[iteration]))
            if abs(current_likelihood - self.likelihoods[iteration]) <= epsilon:
                print ('#'*20 + "EM Found!, the value is: " + str(self.likelihoods[iteration]) + '#'*20)
                break
            else:
                current_likelihood = self.likelihoods[iteration]


def main():
    documents_path = 'C:/Users/ZiZi/OneDrive - University of Illinois - Urbana/CS410 Text Information Sytem/MP3/data/test2.txt'
    #documents_path = 'data/test.txt' #ori
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print('corpus.vocabulary:', corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    
    number_of_topics = 2
    max_iterations = 200
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
