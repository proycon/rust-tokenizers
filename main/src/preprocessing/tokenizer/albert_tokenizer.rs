// Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::preprocessing::vocab::sentence_piece_vocab::{SentencePieceModel, Node};
use crate::{Vocab, Tokenizer, MultiThreadedTokenizer};
use crate::preprocessing::tokenizer::base_tokenizer::{Token, TokenRef, Offset, OffsetSize, Mask};
use crate::tokenization_utils::{clean_text, decompose_nfkc, lowercase, is_whitespace, replace_string};
use crate::preprocessing::tokenizer::tokenization_utils::strip_accents;
use crate::preprocessing::vocab::albert_vocab::AlbertVocab;

pub struct AlbertTokenizer {
    model: SentencePieceModel,
    vocab: AlbertVocab,
    lower_case: bool,
    keep_accents: bool,
}

impl AlbertTokenizer {
    pub fn from_file(path: &str, lower_case: bool, keep_accents: bool) -> AlbertTokenizer {
        let model = SentencePieceModel::from_file(path);
        let vocab = AlbertVocab::from_file(path);
        AlbertTokenizer { model, vocab, lower_case, keep_accents }
    }

    pub fn from_existing_vocab_and_model(vocab: AlbertVocab, model: SentencePieceModel,
                                         lower_case: bool, keep_accents: bool) -> AlbertTokenizer {
        AlbertTokenizer { model, vocab, lower_case, keep_accents }
    }

    fn post_process_pieces<'a>(&self, tokens: &'a mut Vec<Token>) -> &'a Vec<Token> {
        let mut positions_to_update: Vec<(usize, Vec<Token>)> = vec!();
        for (token_idx, token) in tokens.iter().enumerate() {
            let mut token_chars = token.text.chars().rev();
            if token.text.chars().count() > 1 {
                if (token_chars.next().unwrap() == ',') & token_chars.next().unwrap().is_ascii_digit() {
                    let mut new_token = token.clone();
                    let last_char = new_token.text.pop().unwrap();
                    new_token.text = new_token.text.replace('\u{2581}', "");
                    let updated_tokens = self.model.decode_forward_token_ref(new_token.as_ref());
                    let updated_tokens = self.model.decode_backward(&updated_tokens);
                    let mut updated_tokens = self.parse_nodes_to_tokens(updated_tokens);

                    if (token.text.chars().next().unwrap() != '\u{2581}') &
                        (updated_tokens[0].text.chars().next().unwrap() == '\u{2581}') {
                        if updated_tokens[0].text.chars().count() == 1 {
                            updated_tokens.remove(0);
                        } else {
                            let first_char_length = updated_tokens[0].text.chars().next().unwrap().len_utf8();
                            updated_tokens[0].text = (&updated_tokens[0].text[first_char_length..]).parse().unwrap();
                        }
                    }
                    updated_tokens.push(Token {
                        text: last_char.to_string(),
                        offset: Offset { begin: token.offset.end, end: token.offset.end - 1 },
                        reference_offsets: vec!(*token.reference_offsets.last().unwrap()),
                        mask: token.mask,
                    });
                    positions_to_update.push((token_idx, updated_tokens.clone()));
                }
            }
        };
        for (pos, new_tokens) in positions_to_update {
            tokens.splice(pos..pos, new_tokens);
        }
        tokens
    }

    fn parse_nodes_to_tokens(&self, nodes: Vec<&Node>) -> Vec<Token> {
        let mut output: Vec<Token> = Vec::with_capacity(nodes.len() + 1);
        let mut is_prev_unknown = false;
        for node in nodes {
            // Group unknown tokens
            if is_prev_unknown & (node.index == 0) {
                let prev_token = output.last().unwrap();
                let mut text = prev_token.text.clone();
                text.push_str(node.text);
                let mut reference_offsets = prev_token.reference_offsets.clone();
                reference_offsets.extend_from_slice(node.reference_offsets);
                let consolidated_unknown = Token {
                    text,
                    offset: Offset { begin: 0, end: 0 },
                    reference_offsets,
                    mask: Default::default(),
                };
                output.pop();
                output.push(consolidated_unknown);
            } else {
                output.push(Token {
                    text: node.text.to_owned(),
                    offset: Offset { begin: 0, end: 0 },
                    reference_offsets: node.reference_offsets.to_vec(),
                    mask: Default::default(),
                });
            }
            is_prev_unknown = node.index == 0;
        }
        output
    }
}

impl Tokenizer<AlbertVocab> for AlbertTokenizer {
    fn vocab(&self) -> &AlbertVocab { &self.vocab }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        let mut token = text.to_owned();
        replace_string(&mut token, "``", "\"");
        replace_string(&mut token, "\'\'", "\"");
        clean_text(&mut token, true);
        decompose_nfkc(&mut token);
        if self.lower_case {
            lowercase(&mut token);
        }
        if !self.keep_accents {
            strip_accents(&mut token);
        }
        token.text = token.text.replace(|c: char| is_whitespace(&c), "\u{2581}");
        if !token.text.starts_with('\u{2581}') {
            token.text.insert(0, '\u{2581}');
            token.reference_offsets.insert(0, 0);
        };
        let output = self.model.decode_forward_token_ref(token.as_ref());
        let decoded = self.model.decode_backward(&output);

        let mut output: Vec<Token> = self.parse_nodes_to_tokens(decoded);
        self.post_process_pieces(&mut output);
        output
    }


    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.into_iter().map(|v| v.replace('\u{2581}', " ")).collect::<Vec<String>>().join("")
    }

    fn build_input_with_special_tokens(&self, tokens_1: Vec<i64>, tokens_2: Option<Vec<i64>>,
                                       offsets_1: Vec<Option<Offset>>, offsets_2: Option<Vec<Option<Offset>>>,
                                       original_offsets_1: Vec<Vec<OffsetSize>>, original_offsets_2: Option<Vec<Vec<OffsetSize>>>,
                                       mask_1: Vec<Mask>, mask_2: Option<Vec<Mask>>) -> (Vec<i64>, Vec<i8>, Vec<i8>, Vec<Option<Offset>>, Vec<Vec<OffsetSize>>, Vec<Mask>) {
        let mut output: Vec<i64> = vec!();
        let mut token_segment_ids: Vec<i8> = vec!();
        let mut special_tokens_mask: Vec<i8> = vec!();
        let mut offsets: Vec<Option<Offset>> = vec!();
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec!();
        let mut mask: Vec<Mask> = vec!();
        special_tokens_mask.push(1);
        special_tokens_mask.extend(vec![0; tokens_1.len()]);
        special_tokens_mask.push(1);
        token_segment_ids.extend(vec![0; tokens_1.len() + 2]);
        output.push(self.vocab.token_to_id(AlbertVocab::cls_value()));
        output.extend(tokens_1);
        output.push(self.vocab.token_to_id(AlbertVocab::sep_value()));
        offsets.push(None);
        offsets.extend(offsets_1);
        offsets.push(None);
        original_offsets.push(vec!());
        original_offsets.extend(original_offsets_1);
        original_offsets.push(vec!());
        mask.push(Mask::Special);
        mask.extend(mask_1);
        mask.push(Mask::Special);
        if let Some(add_tokens) = tokens_2 {
            let length = add_tokens.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(add_tokens);
            output.push(self.vocab.token_to_id(AlbertVocab::sep_value()));
            if let Some(add_offsets) = offsets_2 {
                offsets.extend(add_offsets);
            } else {
                offsets.extend(vec![None; length]);
            }
            if let Some(add_original_offsets) = original_offsets_2 {
                original_offsets.extend(add_original_offsets);
            }
            offsets.push(None);
            original_offsets.push(vec!());
            if let Some(mask_2) = mask_2 {
                mask.extend(mask_2)
            } else {
                mask.extend(vec![Mask::None; length]);
            }
            mask.push(Mask::Special);
        }
        (output, token_segment_ids, special_tokens_mask, offsets, original_offsets, mask)
    }
}

impl MultiThreadedTokenizer<AlbertVocab> for AlbertTokenizer {}