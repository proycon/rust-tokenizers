use std::path::PathBuf;
use rust_tokenizers::{BertTokenizer, Tokenizer};

fn main() {

    let mut vocab_path = PathBuf::from("E:/Coding/cache/rustbert");
    vocab_path.push("bert");
    vocab_path.push("vocab.txt");

    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);

    let test_input = "Hello [MASK] world!";
    let _test_input2 = "Hello world!";

    let output = tokenizer.tokenize(test_input);

    println!("{:?}", output);

}