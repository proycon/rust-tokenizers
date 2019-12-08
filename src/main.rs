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

//use rust_transformers;
//use rust_transformers::preprocessing::vocab::base_vocab::Vocab;
//use std::process;
//use rust_transformers::preprocessing::adapters::Example;
//use rust_transformers::preprocessing::tokenizer::bert_tokenizer::BertTokenizer;
//use std::time::Instant;
//use rust_transformers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
//use std::sync::Arc;
extern crate tch;

//use tch::Tensor;
//use tch::{Device};

fn main() {
//    let vocab_path = "E:/Coding/rust-transformers/resources/vocab/bert-base-uncased-vocab.txt";
//    let bert_vocab = Arc::new(rust_transformers::BertVocab::from_file(vocab_path));
//
//    let _data = match rust_transformers::preprocessing::adapters::read_sst2(
//        "E:/Coding/rust-transformers/resources/data/SST-2/train.tsv",
//        b'\t') {
//        Ok(examples) => {
//            examples
//        }
//        Err(err) => {
//            println!("{}", err);
//            process::exit(1);
//        }
//    };
//
//    let _text_list: Vec<&str> = _data.iter().map(|v| v.sentence_1.as_ref()).collect();
//    let _before = Instant::now();
//    let _results = bert_tokenizer.encode_list(_text_list, 128, &TruncationStrategy::LongestFirst, 0);
//    println!("Elapsed time: {:.2?}", _before.elapsed());
    let cuda_available = tch::Cuda::cudnn_is_available();
    println!("{}", cuda_available)
//    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]).to(Device::Cuda(0));
//    let t = t * 2;
//    t.print();
}
