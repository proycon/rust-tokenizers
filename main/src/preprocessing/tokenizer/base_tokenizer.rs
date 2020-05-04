// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::preprocessing::vocab::base_vocab::Vocab;
use crate::preprocessing::tokenizer::tokenization_utils::{tokenize_cjk_chars, whitespace_tokenize, strip_accents, split_on_punct, clean_text, truncate_sequences};
use std::sync::Arc;
use rayon::prelude::*;
use itertools::Itertools;

pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
    DoNotTruncate,
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct TokenizedInput {
    pub token_ids: Vec<i64>,
    pub segment_ids: Vec<i8>,
    pub special_tokens_mask: Vec<i8>,
    pub overflowing_tokens: Vec<i64>,
    pub num_truncated_tokens: usize,
}

pub trait Tokenizer<T: Vocab> {
    fn vocab(&self) -> &T;

    fn tokenize(&self, text: &str) -> (Vec<String>, Vec<i64>);

    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<(Vec<String>, Vec<i64>)> {
        text_list.
            into_iter().
            map(|text| self.tokenize(text)).
            collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<i64> {
        tokens.into_iter().map(|v| self.vocab().token_to_id(v)).collect()
    }

    fn encode(&self, text_1: &str, text_2: Option<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> TokenizedInput {
        let (tokens_1, offsets_1) = self.tokenize(text_1);
        let token_ids_1 = self.convert_tokens_to_ids(&tokens_1);
        let len_1 = token_ids_1.len();
        let (token_ids_2, len_2, pair, offsets_2) = {
            if let Some(text) = text_2 {
                let (tokens, offsets) = self.tokenize(text);
                let token_ids_2: Vec<i64> = self.convert_tokens_to_ids(&tokens);
                let len_2 = token_ids_2.len();
                (Some(token_ids_2), len_2, Some(vec!()), offsets)
            } else {
                (None, 0, None, Vec::new())
            }
        };
//        ToDo: handle the offsets update based on truncation
        let (additional_tokens, _, _) = self.build_input_with_special_tokens(vec!(), pair);
        let total_len = len_1 + len_2 + additional_tokens.len();
        let num_truncated_tokens = if total_len > max_len { total_len - max_len } else { 0 };
        let (token_ids_1,
            token_ids_2,
            overflowing_tokens) = truncate_sequences(token_ids_1,
                                                     token_ids_2,
                                                     num_truncated_tokens,
                                                     truncation_strategy,
                                                     stride).unwrap();

        let (token_ids, segment_ids, special_tokens_mask) = self.build_input_with_special_tokens(token_ids_1,
                                                                                                 token_ids_2);

        TokenizedInput { token_ids, segment_ids, special_tokens_mask, overflowing_tokens, num_truncated_tokens }
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .into_iter()
            .map(|text| self.encode(text, None, max_len, truncation_strategy, stride))
            .collect()
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .into_iter()
            .map(|text| self.encode(text.0, Some(text.1), max_len, truncation_strategy, stride))
            .collect()
    }

    fn decode(&self, token_ids: Vec<i64>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> String {
        let tokens: Vec<String> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|id| !self.vocab().special_indices().contains_key(id))
                .map(|id| { self.vocab().id_to_token(id) })
                .collect_vec()
        } else {
            token_ids
                .iter()
                .map(|id| { self.vocab().id_to_token(id) })
                .collect_vec()
        };

        let decoded_string = self.convert_tokens_to_string(tokens);
        if clean_up_tokenization_spaces {
            self.clean_up_tokenization(decoded_string)
        } else {
            decoded_string
        }
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ")
    }

    fn clean_up_tokenization(&self, input_string: String) -> String {
        input_string
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm'", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
    }

    fn decode_list(&self, token_ids_list: Vec<Vec<i64>>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> Vec<String> {
        token_ids_list
            .into_iter()
            .map(|token_ids| self.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces))
            .collect()
    }


    fn build_input_with_special_tokens(&self, mut tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>) -> (Vec<i64>, Vec<i8>, Vec<i8>) {
        let mut token_segment_ids: Vec<i8> = vec![0; tokens_1.len()];
        let mut special_tokens_mask: Vec<i8> = vec![0; tokens_1.len()];

        let output = match tokens_2 {
            Some(tokens) => {
                token_segment_ids.extend(vec![1; tokens.len()]);
                special_tokens_mask.extend(vec![0; tokens.len()]);
                tokens_1.extend(tokens);
                tokens_1
            }
            None => tokens_1
        };
        (output, token_segment_ids, special_tokens_mask)
    }
}

pub trait MultiThreadedTokenizer<T: Vocab>
    where Self: std::marker::Sync + Send + Tokenizer<T> {
    fn vocab(&self) -> &T
    {
        Tokenizer::<T>::vocab(self)
    }

    fn tokenize_list(&self, text_list: Vec<&str>) -> Vec<Vec<String>> {
        text_list.
            par_iter().
            map(|text| self.tokenize(text)).
            collect()
    }

    fn encode_list(&self, text_list: Vec<&str>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .par_iter()
            .map(|text| self.encode(text, None, max_len, truncation_strategy, stride))
            .collect()
    }

    fn encode_pair_list(&self, text_list: Vec<(&str, &str)>, max_len: usize, truncation_strategy: &TruncationStrategy, stride: usize) -> Vec<TokenizedInput> {
        text_list
            .par_iter()
            .map(|text| self.encode(text.0, Some(text.1), max_len, truncation_strategy, stride))
            .collect()
    }

    fn decode_list(&self, token_ids_list: Vec<Vec<i64>>, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> Vec<String> {
        token_ids_list
            .par_iter()
            .map(|token_ids| self.decode(token_ids.to_vec(), skip_special_tokens, clean_up_tokenization_spaces))
            .collect()
    }
}


pub struct BaseTokenizer<T: Vocab> {
    vocab: Arc<T>,
    lower_case: bool,
}

impl<T: Vocab + Sync + Send> BaseTokenizer<T> {
    pub fn from_file(path: &str, lower_case: bool) -> BaseTokenizer<T> {
        let vocab = T::from_file(path);
        BaseTokenizer { vocab: Arc::new(vocab), lower_case }
    }

    pub fn from_existing_vocab(vocab: Arc<T>, lower_case: bool) -> BaseTokenizer<T> {
        BaseTokenizer { vocab, lower_case }
    }
}

impl<T: Vocab + Sync + Send> Tokenizer<T> for BaseTokenizer<T> {
    fn vocab(&self) -> &T {
        &self.vocab
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let tokenized_text: String = tokenize_cjk_chars(clean_text(text, true).as_str());
        let mut tokenized_text: Vec<String> = whitespace_tokenize(tokenized_text.as_str()).into_iter().map(|s| s.to_string()).collect();

        for string in tokenized_text.iter_mut() {
            if !self.vocab.as_ref().special_values().contains_key(string) {
                if self.lower_case {
                    *string = string.to_lowercase();
                }
                *string = strip_accents(string.to_owned());
            }
        }

        let tokenized_text: Vec<String> = tokenized_text
            .into_iter()
            .map(|v| split_on_punct(v, self.vocab.as_ref()))
            .flatten()
            .map(|s| s.to_string())
            .collect();

        let tokenized_text: String = tokenized_text.into_iter().join(" ");
        let tokenized_text: Vec<String> = whitespace_tokenize(tokenized_text.as_str())
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        tokenized_text
    }
}

impl<T: Vocab + Sync + Send> MultiThreadedTokenizer<T> for BaseTokenizer<T> {}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::BertVocab;
    use std::collections::HashMap;
    use crate::preprocessing::vocab::base_vocab::swap_key_values;

    fn generate_test_vocab() -> BertVocab {
        let values: HashMap<String, i64> = [
            ("hello".to_owned(), 0),
            ("world".to_owned(), 1),
            ("[UNK]".to_owned(), 2),
            ("!".to_owned(), 3),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("中".to_owned(), 7),
            ("华".to_owned(), 8),
            ("人".to_owned(), 9),
            ("[PAD]".to_owned(), 10),
            ("una".to_owned(), 11),
            ("##ffa".to_owned(), 12),
            ("##ble".to_owned(), 13)
        ].iter().cloned().collect();

        let special_values: HashMap<String, i64> = [
            ("[UNK]".to_owned(), 2),
            ("[CLS]".to_owned(), 4),
            ("[SEP]".to_owned(), 5),
            ("[MASK]".to_owned(), 6),
            ("[PAD]".to_owned(), 10)
        ].iter().cloned().collect();

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        BertVocab { values, indices, unknown_value: "[UNK]", special_values, special_indices }
    }

    #[test]
    fn test_base_tokenizer() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                vec!("sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "Sentence with [MASK] token.",
                vec!("sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "[CLS]",
                vec!("[CLS]")
            ),
            (
                "[CLS] [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "[CLS]       [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "asdf",
                vec!("asdf")
            ),
            (
                "",
                vec!()
            ),
            (
                "Allons, Flipote, allons; que d'eux je me délivre.",
                vec!("allons", ",", "flipote", ",", "allons", ";", "que", "d", "\'", "eux", "je", "me", "delivre", ".")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中", "华", "人", "民", "共", "和", "国", "[PAD]", "asdf")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(Tokenizer::tokenize_list(&base_tokenizer, source_texts.clone()), expected_results);
        assert_eq!(MultiThreadedTokenizer::tokenize_list(&base_tokenizer, source_texts.clone()), expected_results);
    }

    #[test]
    fn test_no_lower_casing() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, false);
        let test_tuples = [
            (
                "Sentence with [MASK] token.",
                vec!("Sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "Sentence with [MASK] token.",
                vec!("Sentence", "with", "[MASK]", "token", ".")
            ),
            (
                "[CLS]",
                vec!("[CLS]")
            ),
            (
                "[CLS] [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "[CLS]       [PAD]",
                vec!("[CLS]", "[PAD]")
            ),
            (
                "aSdF",
                vec!("aSdF")
            ),
            (
                "",
                vec!()
            ),
            (
                "Allons, Flipote, allons; que d'eux je me délivre.",
                vec!("Allons", ",", "Flipote", ",", "allons", ";", "que", "d", "\'", "eux", "je", "me", "delivre", ".")
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                vec!("[UNK]", "中", "华", "人", "民", "共", "和", "国", "[PAD]", "asdf")
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<Vec<&str>> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.tokenize(*source_text), *expected_result);
        }

        assert_eq!(Tokenizer::tokenize_list(&base_tokenizer, source_texts.clone()), expected_results);
        assert_eq!(MultiThreadedTokenizer::tokenize_list(&base_tokenizer, source_texts.clone()), expected_results);
    }

    #[test]
    fn test_convert_tokens_to_ids() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let test_tuples = [
            (
                vec!("hello", "[MASK]", "world", "!"),
                vec!(0, 6, 1, 3)
            ),
            (
                vec!("hello", ",", "una", "##ffa", "##ble", "world", "!"),
                vec!(0, 2, 11, 12, 13, 1, 3)
            ),
            (
                vec!("[UNK]", "[UNK]", "华", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[UNK]"),
                vec!(2, 2, 8, 2, 2, 2, 2, 2, 10, 2)
            )
        ];

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.convert_tokens_to_ids(source_text.iter().map(|v| String::from(*v)).collect::<Vec<_>>().as_ref()),
                       *expected_result);
        }
    }

    #[test]
    fn test_encode_single_sentence() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
            (
                "hello world!",
                TokenizedInput { token_ids: vec!(0, 1, 3), segment_ids: vec!(0, 0, 0), special_tokens_mask: vec!(0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "hello, unaffable world!",
                TokenizedInput { token_ids: vec!(0, 2, 2, 1, 3), segment_ids: vec!(0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "[UNK]中华人民共和国 [PAD] asdf",
                TokenizedInput { token_ids: vec!(2, 7, 8, 9, 2, 2, 2, 2, 10, 2), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
            (
                "[UNK] a ! c ! e ! g ! i ! [PAD] a ! c ! e ! g ! i !",
                TokenizedInput { token_ids: vec!(2, 2, 3, 2, 3, 2, 3, 2, 3, 2), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(3, 10, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3), num_truncated_tokens: 12 }
            )
        ];
        let source_texts: Vec<&str> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.encode(source_text, None, 10, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(Tokenizer::encode_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_encode_sentence_pair() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let truncation_strategy = TruncationStrategy::LongestFirst;
        let test_tuples = [
//            No truncation required
            (
                ("hello world!", "This is the second sentence"),
                TokenizedInput { token_ids: vec!(0, 1, 3, 2, 2, 2, 2, 2), segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 0 }
            ),
//            Truncation of sentence 2 (longest)
            (
                ("hello world!", "!This is the second sentence!!!"),
                TokenizedInput { token_ids: vec!(0, 1, 3, 3, 2, 2, 2, 2, 2, 3), segment_ids: vec!(0, 0, 0, 1, 1, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(), num_truncated_tokens: 2 }
            ),
//            Truncation of sentence 1 (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello  hello  hello  hello  hello  hello  hello", "!!!"),
                TokenizedInput { token_ids: vec!(2, 0, 0, 0, 0, 0, 0, 3, 3, 3), segment_ids: vec!(0, 0, 0, 0, 0, 0, 0, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(0, 0, 0, 0, 0), num_truncated_tokens: 5 }
            ),
//            Truncation of both sentences (longest)
            (
                ("[UNK] hello  hello  hello  hello  hello", "!!!!!!!!"),
                TokenizedInput { token_ids: vec!(2, 0, 0, 0, 0, 3, 3, 3, 3, 3), segment_ids: vec!(0, 0, 0, 0, 0, 1, 1, 1, 1, 1), special_tokens_mask: vec!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), overflowing_tokens: vec!(0), num_truncated_tokens: 4 }
            )
        ];
        let source_texts: Vec<(&str, &str)> = test_tuples.iter().map(|v| v.0).collect();
        let expected_results: Vec<TokenizedInput> = test_tuples.iter().map(|v| v.1.clone()).collect();

//        When & Then
        for (source_text, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.encode(source_text.0, Some(source_text.1), 10, &truncation_strategy, 0),
                       *expected_result);
        }
        assert_eq!(Tokenizer::encode_pair_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
        assert_eq!(MultiThreadedTokenizer::encode_pair_list(&base_tokenizer, source_texts.clone(), 10, &truncation_strategy, 0), expected_results);
    }

    #[test]
    fn test_decode() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let skip_special_tokens = false;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 3),
                "[PAD] hello world !",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "[PAD] hello world [UNK] !",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_skip_special_tokens() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = false;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 3),
                "hello world !",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "hello world !",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }

    #[test]
    fn test_decode_clean_up_tokenization_spaces() {
//        Given
        let vocab = Arc::new(generate_test_vocab());
        let base_tokenizer: BaseTokenizer<BertVocab> = BaseTokenizer::from_existing_vocab(vocab, true);
        let skip_special_tokens = true;
        let clean_up_tokenization_spaces = true;
        let test_tuples = [
            (
                vec!(0, 1, 3),
                "hello world!",
            ),
            (
                vec!(10, 0, 1, 3),
                "hello world!",
            ),
            (
                vec!(10, 0, 1, 2, 3),
                "hello world!",
            )
        ];
        let source_ids: Vec<Vec<i64>> = test_tuples.iter().map(|v| v.0.clone()).collect_vec();
        let expected_results: Vec<&str> = test_tuples.iter().map(|v| v.1.clone()).collect_vec();

//        When & Then
        for (source_ids, expected_result) in test_tuples.iter() {
            assert_eq!(base_tokenizer.decode(source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces),
                       *expected_result);
        }
        assert_eq!(Tokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
        assert_eq!(MultiThreadedTokenizer::decode_list(&base_tokenizer, source_ids.clone(), skip_special_tokens, clean_up_tokenization_spaces), expected_results);
    }
}
