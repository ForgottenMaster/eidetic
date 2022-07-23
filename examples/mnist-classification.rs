use eidetic::activations::{Linear, ReLU, Tanh};
use eidetic::layers::{Chain, Dense, Dropout, Input};
use eidetic::loss::SoftmaxCrossEntropy;
use eidetic::operations::{
    InitialisedOperation, TrainableOperation, UninitialisedOperation, WithOptimiser,
};
use eidetic::optimisers::learning_rate_handlers::FixedLearningRateHandler;
use eidetic::optimisers::SGDMomentum;
use eidetic::tensors::{rank, Tensor};
use eidetic::training::train;
use eidetic::ElementType;
use mnist::*;
use ndarray::Array;
use std::any::type_name;
use std::fs::{create_dir_all, read, write};
use std::mem::size_of;
use std::path::Path;

// hyperparameters for training.
const LEARNING_RATE: ElementType = 0.001;
const EPOCHS: u16 = 2;
const MOMENTUM: ElementType = 0.9;
const KEEP_PROBABILITY: ElementType = 0.5;
const EVAL_EVERY: u16 = 1;
const BATCH_SIZE: usize = 64;
const SEED: u64 = 42;

fn main() {
    // Read input data as eidetic compatible tensors.
    println!("Reading MNIST input data...");
    let InputData {
        training_images,
        training_labels,
        testing_images,
        testing_labels,
    } = read_mnist_data();

    // Get a trained neural network which will either just initialise
    // one from stored/recorded weights in a file, or will run training with
    // a seed to make and train one.
    println!("Preparing neural network...");
    let network = get_trained_network(
        training_images.clone(),
        training_labels.clone(),
        &testing_images,
        &testing_labels,
    );

    // Get predictions from the trained network for the training images and testing images.
    println!("Obtaining predictions...");
    let training_predictions = network.predict(training_images).unwrap();
    let testing_predictions = network.predict(testing_images).unwrap();

    // Calculate accuracy.
    println!("Calculating accuracy...");
    let training_accuracy = calculate_accuracy(training_predictions, training_labels);
    let testing_accuracy = calculate_accuracy(testing_predictions, testing_labels);
    println!("Accuracy (training): {training_accuracy}%");
    println!("Accuracy (testing): {testing_accuracy}%");
}

fn read_mnist_data() -> InputData {
    // Read the Mnist data which is completely flattened.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .base_path("examples/data/input/mnist")
        .finalize();

    // Convert the input data into tensors which are what the eidetic API uses.
    // Note that most layers expect rank 2 tensors so we'll need to reshape the
    // data we read from file.
    let training_images =
        Tensor::<rank::Two>::new((60_000, 784), trn_img.into_iter().map(ElementType::from))
            .unwrap();
    let training_labels =
        Tensor::<rank::Two>::new((60_000, 10), trn_lbl.into_iter().map(ElementType::from)).unwrap();
    let testing_images =
        Tensor::<rank::Two>::new((10_000, 784), tst_img.into_iter().map(ElementType::from))
            .unwrap();
    let testing_labels =
        Tensor::<rank::Two>::new((10_000, 10), tst_lbl.into_iter().map(ElementType::from)).unwrap();

    // Bundle the tensors up to get the final input data for the example.
    InputData {
        training_images,
        training_labels,
        testing_images,
        testing_labels,
    }
}

fn get_trained_network(
    training_images: Tensor<rank::Two>,
    training_labels: Tensor<rank::Two>,
    testing_images: &Tensor<rank::Two>,
    testing_labels: &Tensor<rank::Two>,
) -> impl InitialisedOperation<Input = Tensor<rank::Two>, Output = Tensor<rank::Two>> {
    // The structure of the uninitialised network doesn't depend on whether we've recorded
    // the weights previously or not.
    let network = Input::new(784)
        .chain(Dense::new(300, Tanh::new()))
        .chain(Dropout::new(KEEP_PROBABILITY))
        .chain(Dense::new(100, ReLU::new()))
        .chain(Dense::new(10, Linear::new()));

    // Get the path for the trained weights.
    let dir_path = Path::new("examples/data/output/mnist-classification");
    let file_path_string = format!("weights-{}.bin", type_name::<ElementType>());
    let file_path = dir_path.join(file_path_string);
    println!(
        "Looking for recorded weights in file at path: {}...",
        file_path.display()
    );

    // If a file at that path exists, then just use the data inside of the file as
    // the weights for the neural network. Otherwise, we'll want to run full training on the
    // network and stash the weights after training.
    if file_path.exists() {
        println!("Using recorded weights...");
        let bytes = read(file_path).unwrap();
        let weights = bytes
            .chunks(size_of::<ElementType>())
            .map(|chunk| ElementType::from_be_bytes(chunk.try_into().unwrap()));
        network.with_iter(weights).unwrap()
    } else {
        println!("Training a new neural network from seed ({SEED})...");
        let network = train(
            network.with_seed(SEED).with_optimiser(SGDMomentum::new(
                FixedLearningRateHandler::new(LEARNING_RATE),
                MOMENTUM,
            )),
            &SoftmaxCrossEntropy::new(),
            training_images,
            training_labels,
            &testing_images,
            &testing_labels,
            EPOCHS,
            EVAL_EVERY,
            BATCH_SIZE,
            SEED,
        )
        .unwrap()
        .into_initialised();
        println!("Saving trained weights into file...");
        let bytes = network
            .iter()
            .flat_map(|elem| elem.to_be_bytes().into_iter())
            .collect::<Vec<_>>();
        create_dir_all(dir_path).unwrap();
        write(file_path, bytes).unwrap();
        network
    }
}

fn calculate_accuracy(predictions: Tensor<rank::Two>, targets: Tensor<rank::Two>) -> ElementType {
    let cols = 10;
    let predictions = Array::from_iter(predictions.into_iter());
    let rows = predictions.len() / cols;
    let predictions = predictions.into_shape((rows, cols)).unwrap();
    let targets = Array::from_iter(targets.into_iter())
        .into_shape((rows, cols))
        .unwrap();
    let correct_count = predictions
        .rows()
        .into_iter()
        .zip(targets.rows().into_iter())
        .filter(|(prediction, target)| {
            let mut predicted_max = ElementType::MIN;
            let mut predicted_number = 0;
            prediction.iter().enumerate().for_each(|(idx, value)| {
                if *value > predicted_max {
                    predicted_max = *value;
                    predicted_number = idx;
                }
            });
            let target_number = target
                .iter()
                .enumerate()
                .filter(|(_, value)| **value == 1.0)
                .map(|(idx, _)| idx)
                .next()
                .unwrap();
            predicted_number == target_number
        })
        .count();
    let total_count = predictions.nrows();
    ((correct_count as ElementType) / (total_count as ElementType)) * 100.0
}

struct InputData {
    training_images: Tensor<rank::Two>,
    training_labels: Tensor<rank::Two>,
    testing_images: Tensor<rank::Two>,
    testing_labels: Tensor<rank::Two>,
}
