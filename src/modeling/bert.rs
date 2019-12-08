// Copyright 2018 The Google AI Language Team Authors
// Copyright 2018 2018, NVIDIA CORPORATION
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

extern crate tch;

use tch::Tensor;

fn gelu(x: Tensor) -> Tensor {
    &x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf())
}

//==============================
// Unit tests
//==============================
#[cfg(test)]
mod tests {
    use super::*;
    use super::tch::kind::Kind::Float;

    #[test]
    fn test_gelu() {
//        Given
        let x = Tensor::of_slice(&[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]);
        let expected_output = &[0.0, 0.0, 0.0, -1.2672e-04, -4.55e-02,
            0.0, 1.9545, 3.9999, 6.0, 8.0, 10.0];
        let epsilon = 0.001;

//        When
        let test_result = gelu(x);

//        Then
        let diff: Tensor = &test_result - Tensor::of_slice(expected_output);
        let output = f32::from(diff.sum(Float)).abs();
        assert!(output <= epsilon);
    }
}
