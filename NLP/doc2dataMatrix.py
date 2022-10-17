import re
import string
class DocumentDict(object):
    def __init__(self,name, document_string):
        self.document_string = document_string
        self.name = name
        self.para = []
        self.sent = []
        self.words = []
        
                
    def MetaTags(self):
        meta_data = {'meta':{'para':self.paragraph(), 
                                'sent':self.sentence(), 
                                'words': self.allwords(),
                               }
                    }
        return meta_data
    
    def paragraph(self):
        para_exp2 = []
        exp2_index = 0
        exp2_split_end = 0
        for match in re.finditer(r'(?s)((?:[^\n][\n]?)+)', self.document_string):
            para_exp2.append(self.document_string[match.start():match.end()])
        for para in para_exp2:
            start = exp2_split_end
            end = len(para)
            address = (exp2_index, start, start + end)
            self.para.append(address)
            exp2_split_end = start + end
            exp2_index += 1
        return self.para
    
    
    def sentence(self):
        sent_exp = []
        sent_exp_index = 0
        sent_exp_end = 0
        for match in re.finditer(r' *[\.\?!][\'"\)\]]* *', self.document_string):
            sent_exp.append(self.document_string[match.start():match.end()])
        for sent in sent_exp:
            start = sent_exp_end
            end = len(sent)
            address = (sent_exp_index, start, start + end)
            self.sent.append(address)
            sent_exp_end = start + end
            sent_exp_index += 1
        return self.sent
    
    def _sentence(document):
        _sent = []
        sent_exp = []
        sent_exp_index = 0
        sent_exp_end = 0
        #print('5 printing document in _sentence', document)
        #print('8 printing document type in _sentence', type(document))
        sent = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)', document)
        for sentence in sent:
            sent_exp.append(sentence)
            #print('5 sentence in sentence for loop', sentence)
#         for match in re.finditer(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)', document):
#             print('9a printing match in _sentence', match)
#             sent = document[match.start():match.end()]
#             print('9 sent print in _sentence', sent)
#             sent_exp.append(sent)

        for sent in sent_exp:
            #print('6 _sentence sent', sent)
            start = sent_exp_end
            end = len(sent)
            address = (sent_exp_index, start, start + end)
            _sent.append(address)
            sent_exp_end = start + end
            sent_exp_index += 1
        return _sent

    
    def allwords(self):
        word_exp = []
        word_exp_index = 0
        word_exp_end = 0
        for match in re.finditer(r'\b\w+\b', self.document_string):
            word_exp.append(self.document_string[match.start():match.end()])
        for word in word_exp:
            start = word_exp_end
            end = len(word)
            address = (word_exp_index, start, start + end)
            self.words.append(address)
            word_exp_end = start + end
            word_exp_index += 1
        return self.words
    
    def _allwords(document):
        _words = []
        word_exp = []
        word_exp_index = 0
        word_exp_end = 0
        #print('11 print type {}, length {} and document {}  in _allwords'.format(type(document), len(document), document))
        for match in re.finditer(r'\w+', document):
            #print(match)
            #print('12 print word match', match)
            #print('printing words in _all', document[match.start():match.end()])
            word_exp.append(document[match.start():match.end()])
#             word = match.group()
#             word_exp.append(word)
            
#         res = re.sub('['+string.punctuation+']', '', document).split()
#         for word in res:
#             word_exp.append(word)
    
        for word in word_exp:
            start = word_exp_end
            end = len(word)
            word_address = (word_exp_index, start, start + end, word)
            _words.append(word_address)
            word_exp_end = start + end
            word_exp_index += 1
            #print('13 printing return in _allwords ', _words)
        return _words
    
    def meta_data_matrix(self):
        # applying deterministic approach using regular expression
        meta_data_matrix = []
        para_id = 0
        sent_id = 0
        word_id = 0
        meta_matrix = []
        para = self.paragraph()
#         print('1 length of para', len(para))
#         print('2 print para', para)
        while para_id <= len(para) -1:
            #para_text = self.document_string[para[para_id][para[1]]:para[para_id][para[2]]]
            para_text = self.document_string[para[para_id][1]:para[para_id][2]]
#             print('3 para_text type {} and para_text len {}'.format(type(para_text), len(para_text)))
#             print('4 printing para text',para_text)
            sent = DocumentDict._sentence(para_text)
            sent_count = len(sent)
#             print('7 printing sent type {} sent len {} '.format(type(sent), len(sent)))
#             print('7b print sentence', sent)
            while sent_id <= sent_count -1:
                #print('8 printing sent_id', sent_id)
                #print('9 printing sent', sent)
                sent_text = para_text[sent[sent_id][1]:sent[sent_id][2]]
                print('10 printint sent_text', sent_text)
                words = DocumentDict._allwords(sent_text)
                if len(words) > 0:
                    while word_id <= len(words) -1:
                        #print('14 words length:{}, type:{}, word:{}'.format(len(words),type(words), words))
                        #word = sent_text[word_id[1]:word_id[2]]
                        #print('15 printing word_id ',word_id)
                        #word = sent_text[words[word_id][1]:words[word_id][2]]
                        
                        #print('16 printing word in ', word)
                        matrix = [para_id, sent_id, word_id, words[word_id][3]]
                        #print('17 meta matrix')
                        meta_matrix.append(matrix)
                        word_id +=1
                        #print('18 incrementing word_id', word_id)
                else:
                    pass
                sent_id +=1
                #print('19 incrementing sent_id', sent_id)
            para_id +=1
            #print('20 printing para_id', para_id)
        return meta_matrix
    
    def data_matrix(self):
        pass
        