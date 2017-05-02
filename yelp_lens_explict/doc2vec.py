import gensim
import numpy as np
import logging
import multiprocessing

NUM_WORKERS = multiprocessing.cpu_count()

class corpusIterable(object):
    def __iter__(self):
        """
        The iterable interface: return an iterator from __iter__().

        Every generator is an iterator implicitly (but not vice versa!),
        so implementing `__iter__` as a generator is the easiest way
        to create streamed iterables.
        
        for more on iterators vs generators
        https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

        """
        with open('../data/yelp/yelp_pruned_20.dat') as f:
            for index, line in enumerate(f):
                data = line.split("::", 4)
                # For training data, add tags
                # check https://datascience.stackexchange.com/questions/10216/doc2vec-how-to-label-the-paragraphs-gensim
                # for a description of the labels
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(data[4]), [str(index)])


def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        for index,line in enumerate(f):
            data = line.split("::", 4)
            if tokens_only:
                yield gensim.utils.simple_preprocess(data[4])
            else:
                # For training data, add tags
                # check https://datascience.stackexchange.com/questions/10216/doc2vec-how-to-label-the-paragraphs-gensim
                # for a description of the labels
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(data[4]), [str(index)])

def doc2vec(input_file_path, output_file_path):
    """
    runs gensim doc2vec on the textual ratings embedded within file.
    :param input_file_path: a file in the form user, restaurant, rating, review
    :return: saves a new npy file with a vector instead of text
    """
    iterable = corpusIterable()
    logging.basicConfig( level=logging.INFO)
    logger = logging.getLogger(__name__)
    #train_corpus = list(read_corpus(input_file_path))
    logger.info('corpus built')
    model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=2, iter=5, workers=NUM_WORKERS)
    model.build_vocab(iterable)
    logger.info('vocab built')
    model.train(iterable, total_examples=model.corpus_count)
    logger.info('model trained')
    np.save(output_file_path, np.matrix(model.docvecs))


if __name__ == '__main__':

    doc2vec('../data/yelp/yelp_pruned_20.dat', 'input/docvecs.npy')


