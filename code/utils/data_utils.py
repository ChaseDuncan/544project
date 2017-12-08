from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions

datapath="../data/iwslt15-test/"
ftraindata=datapath+"train.vi"
etraindata=datapath+"train.en"
fevaldata=datapath+"tst2012.vi"
eevaldata=datapath+"tst2012.en"

ds = Dataset('iwslt15-vi2en-test', datapath, silence=False)

ds.setOutput(etraindata,
             'train',
             type='text',
             id='target_text',
             tokenization='tokenize_none',
             build_vocabulary=True,
             pad_on_batch=True,
             sample_weights=True,
             max_text_len=50,
             max_words=30000,
             min_occ=0)

ds.setOutput(eevaldata,
             'val',
             type='text',
             id='target_text',
             pad_on_batch=True,
             tokenization='tokenize_none',
             sample_weights=True,
             max_text_len=50,
             max_words=0)

ds.setInput(ftraindata,
            'train',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            build_vocabulary=True,
            fill='end',
            max_text_len=50,
            max_words=30000,
            min_occ=0)

ds.setInput(fevaldata,
            'val',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            fill='end',
            max_text_len=50,
            min_occ=0)

# offset by one for 'teacher forcing'
# ref:
# https://web.stanford.edu/class/psych209a/ReadingsByDate/02_25/Williams%20Zipser95RecNets.pdf
ds.setInput(etraindata,
            'train',
            type='text',
            id='state_below',
            required=False,
            tokenization='tokenize_none',
            pad_on_batch=True,
            build_vocabulary='target_text',
            offset=1,
            fill='end',
            max_text_len=50,
            max_words=30000)

ds.setInput(None,
            'val',
            type='ghost',
            id='state_below',
            required=False)

keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

saveDataset(ds, 'datasets')



