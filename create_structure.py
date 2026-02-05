import os
import json

# D2L kitap yapısı - verilen içerik listesine göre
structure = {
    "1_introduction": {
        "1.1_a-motivating-example": [],
        "1.2_key-components": [],
        "1.3_kinds-of-machine-learning-problems": [],
        "1.4_roots": [],
        "1.5_the-road-to-deep-learning": [],
        "1.6_success-stories": [],
        "1.7_the-essence-of-deep-learning": [],
        "1.8_summary": [],
        "1.9_exercises": []
    },
    "2_preliminaries": {
        "2.1_data-manipulation": [
            "2.1.1_getting-started",
            "2.1.2_indexing-and-slicing",
            "2.1.3_operations",
            "2.1.4_broadcasting",
            "2.1.5_saving-memory",
            "2.1.6_conversion-to-other-python-objects",
            "2.1.7_summary",
            "2.1.8_exercises"
        ],
        "2.2_data-preprocessing": [
            "2.2.1_reading-the-dataset",
            "2.2.2_data-preparation",
            "2.2.3_conversion-to-the-tensor-format",
            "2.2.4_discussion",
            "2.2.5_exercises"
        ],
        "2.3_linear-algebra": [
            "2.3.1_scalars",
            "2.3.2_vectors",
            "2.3.3_matrices",
            "2.3.4_tensors",
            "2.3.5_basic-properties-of-tensor-arithmetic",
            "2.3.6_reduction",
            "2.3.7_non-reduction-sum",
            "2.3.8_dot-products",
            "2.3.9_matrix-vector-products",
            "2.3.10_matrix-matrix-multiplication",
            "2.3.11_norms",
            "2.3.12_discussion",
            "2.3.13_exercises"
        ],
        "2.4_calculus": [
            "2.4.1_derivatives-and-differentiation",
            "2.4.2_visualization-utilities",
            "2.4.3_partial-derivatives-and-gradients",
            "2.4.4_chain-rule",
            "2.4.5_discussion",
            "2.4.6_exercises"
        ],
        "2.5_automatic-differentiation": [
            "2.5.1_a-simple-function",
            "2.5.2_backward-for-non-scalar-variables",
            "2.5.3_detaching-computation",
            "2.5.4_gradients-and-python-control-flow",
            "2.5.5_discussion",
            "2.5.6_exercises"
        ],
        "2.6_probability-and-statistics": [
            "2.6.1_a-simple-example-tossing-coins",
            "2.6.2_a-more-formal-treatment",
            "2.6.3_random-variables",
            "2.6.4_multiple-random-variables",
            "2.6.5_an-example",
            "2.6.6_expectations",
            "2.6.7_discussion",
            "2.6.8_exercises"
        ],
        "2.7_documentation": [
            "2.7.1_functions-and-classes-in-a-module",
            "2.7.2_specific-functions-and-classes"
        ]
    },
    "3_linear-neural-networks-for-regression": {
        "3.1_linear-regression": [
            "3.1.1_basics",
            "3.1.2_vectorization-for-speed",
            "3.1.3_the-normal-distribution-and-squared-loss",
            "3.1.4_linear-regression-as-a-neural-network",
            "3.1.5_summary",
            "3.1.6_exercises"
        ],
        "3.2_object-oriented-design-for-implementation": [
            "3.2.1_utilities",
            "3.2.2_models",
            "3.2.3_data",
            "3.2.4_training",
            "3.2.5_summary",
            "3.2.6_exercises"
        ],
        "3.3_synthetic-regression-data": [
            "3.3.1_generating-the-dataset",
            "3.3.2_reading-the-dataset",
            "3.3.3_concise-implementation-of-the-data-loader",
            "3.3.4_summary",
            "3.3.5_exercises"
        ],
        "3.4_linear-regression-implementation-from-scratch": [
            "3.4.1_defining-the-model",
            "3.4.2_defining-the-loss-function",
            "3.4.3_defining-the-optimization-algorithm",
            "3.4.4_training",
            "3.4.5_summary",
            "3.4.6_exercises"
        ],
        "3.5_concise-implementation-of-linear-regression": [
            "3.5.1_defining-the-model",
            "3.5.2_defining-the-loss-function",
            "3.5.3_defining-the-optimization-algorithm",
            "3.5.4_training",
            "3.5.5_summary",
            "3.5.6_exercises"
        ],
        "3.6_generalization": [
            "3.6.1_training-error-and-generalization-error",
            "3.6.2_underfitting-or-overfitting",
            "3.6.3_model-selection",
            "3.6.4_summary",
            "3.6.5_exercises"
        ],
        "3.7_weight-decay": [
            "3.7.1_norms-and-weight-decay",
            "3.7.2_high-dimensional-linear-regression",
            "3.7.3_implementation-from-scratch",
            "3.7.4_concise-implementation",
            "3.7.5_summary",
            "3.7.6_exercises"
        ]
    },
    "4_linear-neural-networks-for-classification": {
        "4.1_softmax-regression": [
            "4.1.1_classification",
            "4.1.2_loss-function",
            "4.1.3_information-theory-basics",
            "4.1.4_summary-and-discussion",
            "4.1.5_exercises"
        ],
        "4.2_the-image-classification-dataset": [
            "4.2.1_loading-the-dataset",
            "4.2.2_reading-a-minibatch",
            "4.2.3_visualization",
            "4.2.4_summary",
            "4.2.5_exercises"
        ],
        "4.3_the-base-classification-model": [
            "4.3.1_the-classifier-class",
            "4.3.2_accuracy",
            "4.3.3_summary",
            "4.3.4_exercises"
        ],
        "4.4_softmax-regression-implementation-from-scratch": [
            "4.4.1_the-softmax",
            "4.4.2_the-model",
            "4.4.3_the-cross-entropy-loss",
            "4.4.4_training",
            "4.4.5_prediction",
            "4.4.6_summary",
            "4.4.7_exercises"
        ],
        "4.5_concise-implementation-of-softmax-regression": [
            "4.5.1_defining-the-model",
            "4.5.2_softmax-revisited",
            "4.5.3_training",
            "4.5.4_summary",
            "4.5.5_exercises"
        ],
        "4.6_generalization-in-classification": [
            "4.6.1_the-test-set",
            "4.6.2_test-set-reuse",
            "4.6.3_statistical-learning-theory",
            "4.6.4_summary",
            "4.6.5_exercises"
        ],
        "4.7_environment-and-distribution-shift": [
            "4.7.1_types-of-distribution-shift",
            "4.7.2_examples-of-distribution-shift",
            "4.7.3_correction-of-distribution-shift",
            "4.7.4_a-taxonomy-of-learning-problems",
            "4.7.5_fairness-accountability-and-transparency-in-machine-learning",
            "4.7.6_summary",
            "4.7.7_exercises"
        ]
    },
    "5_multilayer-perceptrons": {
        "5.1_multilayer-perceptrons": [
            "5.1.1_hidden-layers",
            "5.1.2_activation-functions",
            "5.1.3_summary-and-discussion",
            "5.1.4_exercises"
        ],
        "5.2_implementation-of-multilayer-perceptrons": [
            "5.2.1_implementation-from-scratch",
            "5.2.2_concise-implementation",
            "5.2.3_summary",
            "5.2.4_exercises"
        ],
        "5.3_forward-propagation-backward-propagation-and-computational-graphs": [
            "5.3.1_forward-propagation",
            "5.3.2_computational-graph-of-forward-propagation",
            "5.3.3_backpropagation",
            "5.3.4_training-neural-networks",
            "5.3.5_summary",
            "5.3.6_exercises"
        ],
        "5.4_numerical-stability-and-initialization": [
            "5.4.1_vanishing-and-exploding-gradients",
            "5.4.2_parameter-initialization",
            "5.4.3_summary",
            "5.4.4_exercises"
        ],
        "5.5_generalization-in-deep-learning": [
            "5.5.1_revisiting-overfitting-and-regularization",
            "5.5.2_inspiration-from-nonparametrics",
            "5.5.3_early-stopping",
            "5.5.4_classical-regularization-methods-for-deep-networks",
            "5.5.5_summary",
            "5.5.6_exercises"
        ],
        "5.6_dropout": [
            "5.6.1_dropout-in-practice",
            "5.6.2_implementation-from-scratch",
            "5.6.3_concise-implementation",
            "5.6.4_summary",
            "5.6.5_exercises"
        ],
        "5.7_predicting-house-prices-on-kaggle": [
            "5.7.1_downloading-data",
            "5.7.2_kaggle",
            "5.7.3_accessing-and-reading-the-dataset",
            "5.7.4_data-preprocessing",
            "5.7.5_error-measure",
            "5.7.6_k-fold-cross-validation",
            "5.7.7_model-selection",
            "5.7.8_submitting-predictions-on-kaggle",
            "5.7.9_summary-and-discussion",
            "5.7.10_exercises"
        ]
    },
    "6_builders-guide": {
        "6.1_layers-and-modules": [
            "6.1.1_a-custom-module",
            "6.1.2_the-sequential-module",
            "6.1.3_executing-code-in-the-forward-propagation-method",
            "6.1.4_summary",
            "6.1.5_exercises"
        ],
        "6.2_parameter-management": [
            "6.2.1_parameter-access",
            "6.2.2_tied-parameters",
            "6.2.3_summary",
            "6.2.4_exercises"
        ],
        "6.3_parameter-initialization": [
            "6.3.1_built-in-initialization",
            "6.3.2_summary",
            "6.3.3_exercises"
        ],
        "6.4_lazy-initialization": [
            "6.4.1_summary",
            "6.4.2_exercises"
        ],
        "6.5_custom-layers": [
            "6.5.1_layers-without-parameters",
            "6.5.2_layers-with-parameters",
            "6.5.3_summary",
            "6.5.4_exercises"
        ],
        "6.6_file-io": [
            "6.6.1_loading-and-saving-tensors",
            "6.6.2_loading-and-saving-model-parameters",
            "6.6.3_summary",
            "6.6.4_exercises"
        ],
        "6.7_gpus": [
            "6.7.1_computing-devices",
            "6.7.2_tensors-and-gpus",
            "6.7.3_neural-networks-and-gpus",
            "6.7.4_summary",
            "6.7.5_exercises"
        ]
    },
    "7_convolutional-neuraş-networks": {
        "7.1_from-fully-connected-layers-to-convolutions": [
            "7.1.1_invariance",
            "7.1.2_constraining-the-mlp",
            "7.1.3_convolutions",
            "7.1.4_channels",
            "7.1.5_summary-and-discussion",
            "7.1.6_exercises"
        ],
        "7.2_convolutions-for-images": [
            "7.2.1_the-cross-correlation-operation",
            "7.2.2_convolutional-layers",
            "7.2.3_object-edge-detection-in-images",
            "7.2.4_learning-a-kernel",
            "7.2.5_cross-correlation-and-convolution",
            "7.2.6_feature-map-and-receptive-field",
            "7.2.7_summary",
            "7.2.8_exercises"
        ],
        "7.3_padding-and-stride": [
            "7.3.1_padding",
            "7.3.2_stride",
            "7.3.3_summary-and-discussion",
            "7.3.4_exercises"
        ],
        "7.4_multiple-input-and-multiple-output-channels": [
            "7.4.1_multiple-input-channels",
            "7.4.2_multiple-output-channels",
            "7.4.3_1x1-convolutional-layer",
            "7.4.4_discussion",
            "7.4.5_exercises"
        ],
        "7.5_pooling": [
            "7.5.1_maximum-pooling-and-average-pooling",
            "7.5.2_padding-and-stride",
            "7.5.3_multiple-channels",
            "7.5.4_summary",
            "7.5.5_exercises"
        ],
        "7.6_convolutional-neural-networks-lenet": [
            "7.6.1_lenet",
            "7.6.2_training",
            "7.6.3_summary",
            "7.6.4_exercises"
        ]
    },
    "8_modern-convolutional-neural-networks": {
        "8.1_deep-convolutional-neural-networks-alexnet": [
            "8.1.1_representation-learning",
            "8.1.2_alexnet",
            "8.1.3_training",
            "8.1.4_discussion",
            "8.1.5_exercises"
        ],
        "8.2_networks-using-blocks-vgg": [
            "8.2.1_vgg-blocks",
            "8.2.2_vgg-network",
            "8.2.3_training",
            "8.2.4_summary",
            "8.2.5_exercises"
        ],
        "8.3_network-in-network-nin": [
            "8.3.1_nin-blocks",
            "8.3.2_nin-model",
            "8.3.3_training",
            "8.3.4_summary",
            "8.3.5_exercises"
        ],
        "8.4_multi-branch-networks-googlenet": [
            "8.4.1_inception-blocks",
            "8.4.2_googlenet-model",
            "8.4.3_training",
            "8.4.4_discussion",
            "8.4.5_exercises"
        ],
        "8.5_batch-normalization": [
            "8.5.1_training-deep-networks",
            "8.5.2_batch-normalization-layers",
            "8.5.3_implementation-from-scratch",
            "8.5.4_lenet-with-batch-normalization",
            "8.5.5_concise-implementation",
            "8.5.6_discussion",
            "8.5.7_exercises"
        ],
        "8.6_residual-networks-resnet-and-resnext": [
            "8.6.1_function-classes",
            "8.6.2_residual-blocks",
            "8.6.3_resnet-model",
            "8.6.4_training",
            "8.6.5_resnext",
            "8.6.6_summary-and-discussion",
            "8.6.7_exercises"
        ],
        "8.7_densely-connected-networks-densenet": [
            "8.7.1_from-resnet-to-densenet",
            "8.7.2_dense-blocks",
            "8.7.3_transition-layers",
            "8.7.4_densenet-model",
            "8.7.5_training",
            "8.7.6_summary-and-discussion",
            "8.7.7_exercises"
        ],
        "8.8_designing-convolution-network-architectures": [
            "8.8.1_the-anynet-design-space",
            "8.8.2_distributions-and-parameters-of-design-spaces",
            "8.8.3_regnet",
            "8.8.4_training",
            "8.8.5_discussion",
            "8.8.6_exercises"
        ]
    },
    "9_recurrent-neural-networks": {
        "9.1_working-with-sequences": [
            "9.1.1_autoregressive-models",
            "9.1.2_sequence-models",
            "9.1.3_training",
            "9.1.4_prediction",
            "9.1.5_summary",
            "9.1.6_exercises"
        ],
        "9.2_converting-raw-text-into-sequence-data": [
            "9.2.1_reading-the-dataset",
            "9.2.2_tokenization",
            "9.2.3_vocabulary",
            "9.2.4_putting-it-all-together",
            "9.2.5_exploratory-language-statistics",
            "9.2.6_summary",
            "9.2.7_exercises"
        ],
        "9.3_language-models": [
            "9.3.1_learning-language-models",
            "9.3.2_perplexity",
            "9.3.3_partitioning-sequences",
            "9.3.4_summary-and-discussion",
            "9.3.5_exercises"
        ],
        "9.4_recurrent-neural-networks": [
            "9.4.1_neural-networks-without-hidden-states",
            "9.4.2_recurrent-neural-networks-with-hidden-states",
            "9.4.3_rnn-based-character-level-language-models",
            "9.4.4_summary",
            "9.4.5_exercises"
        ],
        "9.5_recurrent-neural-network-implementation-from-scratch": [
            "9.5.1_rnn-model",
            "9.5.2_rnn-based-language-model",
            "9.5.3_gradient-clipping",
            "9.5.4_training",
            "9.5.5_decoding",
            "9.5.6_summary",
            "9.5.7_exercises"
        ],
        "9.6_concise-implementation-of-recurrent-neural-networks": [
            "9.6.1_defining-the-model",
            "9.6.2_training-and-predicting",
            "9.6.3_summary",
            "9.6.4_exercises"
        ],
        "9.7_backpropagation-through-time": [
            "9.7.1_analysis-of-gradients-in-rnns",
            "9.7.2_backpropagation-through-time-in-detail",
            "9.7.3_summary",
            "9.7.4_exercises"
        ]
    },
    "10_modern-recurrent-neural-networks": {
        "10.1_long-short-term-memory-lstm": [
            "10.1.1_gated-memory-cell",
            "10.1.2_implementation-from-scratch",
            "10.1.3_concise-implementation",
            "10.1.4_summary",
            "10.1.5_exercises"
        ],
        "10.2_gated-recurrent-units-gru": [
            "10.2.1_reset-gate-and-update-gate",
            "10.2.2_candidate-hidden-state",
            "10.2.3_hidden-state",
            "10.2.4_implementation-from-scratch",
            "10.2.5_concise-implementation",
            "10.2.6_summary",
            "10.2.7_exercises"
        ],
        "10.3_deep-recurrent-neural-networks": [
            "10.3.1_implementation-from-scratch",
            "10.3.2_concise-implementation",
            "10.3.3_summary",
            "10.3.4_exercises"
        ],
        "10.4_bidirectional-recurrent-neural-networks": [
            "10.4.1_implementation-from-scratch",
            "10.4.2_concise-implementation",
            "10.4.3_summary",
            "10.4.4_exercises"
        ],
        "10.5_machine-translation-and-the-dataset": [
            "10.5.1_downloading-and-preprocessing-the-dataset",
            "10.5.2_tokenization",
            "10.5.3_loading-sequences-of-fixed-length",
            "10.5.4_reading-the-dataset",
            "10.5.5_summary",
            "10.5.6_exercises"
        ],
        "10.6_the-encoder-decoder-architecture": [
            "10.6.1_encoder",
            "10.6.2_decoder",
            "10.6.3_putting-the-encoder-and-decoder-together",
            "10.6.4_summary",
            "10.6.5_exercises"
        ],
        "10.7_sequence-to-sequence-learning-for-machine-translation": [
            "10.7.1_teacher-forcing",
            "10.7.2_encoder",
            "10.7.3_decoder",
            "10.7.4_encoder-decoder-for-sequence-to-sequence-learning",
            "10.7.5_loss-function-with-masking",
            "10.7.6_training",
            "10.7.7_prediction",
            "10.7.8_evaluation-of-predicted-sequences",
            "10.7.9_summary",
            "10.7.10_exercises"
        ],
        "10.8_beam-search": [
            "10.8.1_greedy-search",
            "10.8.2_exhaustive-search",
            "10.8.3_beam-search",
            "10.8.4_summary",
            "10.8.5_exercises"
        ]
    },
    "11_attention-mechanism-and-transformers": {
        "11.1_queries-keys-and-values": [
            "11.1.1_visualization",
            "11.1.2_summary",
            "11.1.3_exercises"
        ],
        "11.2_attention-pooling-by-similarity": [
            "11.2.1_kernels-and-data",
            "11.2.2_attention-pooling-via-nadaraya-watson-regression",
            "11.2.3_adapting-attention-pooling",
            "11.2.4_summary",
            "11.2.5_exercises"
        ],
        "11.3_attention-scoring-functions": [
            "11.3.1_dot-product-attention",
            "11.3.2_convenience-functions",
            "11.3.3_scaled-dot-product-attention",
            "11.3.4_additive-attention",
            "11.3.5_summary",
            "11.3.6_exercises"
        ],
        "11.4_the-bahdanau-attention-mechanism": [
            "11.4.1_model",
            "11.4.2_defining-the-decoder-with-attention",
            "11.4.3_training",
            "11.4.4_summary",
            "11.4.5_exercises"
        ],
        "11.5_multi-head-attention": [
            "11.5.1_model",
            "11.5.2_implementation",
            "11.5.3_summary",
            "11.5.4_exercises"
        ],
        "11.6_self-attention-and-positional-encoding": [
            "11.6.1_self-attention",
            "11.6.2_comparing-cnns-rnns-and-self-attention",
            "11.6.3_positional-encoding",
            "11.6.4_summary",
            "11.6.5_exercises"
        ],
        "11.7_the-transformer-architecture": [
            "11.7.1_model",
            "11.7.2_positionwise-feed-forward-networks",
            "11.7.3_residual-connection-and-layer-normalization",
            "11.7.4_encoder",
            "11.7.5_decoder",
            "11.7.6_training",
            "11.7.7_summary",
            "11.7.8_exercises"
        ],
        "11.8_transformers-for-vision": [
            "11.8.1_model",
            "11.8.2_patch-embedding",
            "11.8.3_vision-transformer-encoder",
            "11.8.4_putting-it-all-together",
            "11.8.5_training",
            "11.8.6_summary-and-discussion",
            "11.8.7_exercises"
        ],
        "11.9_large-scale-pretraining-with-transformers": [
            "11.9.1_encoder-only",
            "11.9.2_encoder-decoder",
            "11.9.3_decoder-only",
            "11.9.4_scalability",
            "11.9.5_large-language-models",
            "11.9.6_summary-and-discussion",
            "11.9.7_exercises"
        ]
    },
    "12_optimization-algorithms": {
        "12.1_optimization-and-deep-learning": [
            "12.1.1_goal-of-optimization",
            "12.1.2_optimization-challenges-in-deep-learning",
            "12.1.3_summary",
            "12.1.4_exercises"
        ],
        "12.2_convexity": [
            "12.2.1_definitions",
            "12.2.2_properties",
            "12.2.3_constraints",
            "12.2.4_summary",
            "12.2.5_exercises"
        ],
        "12.3_gradient-descent": [
            "12.3.1_one-dimensional-gradient-descent",
            "12.3.2_multivariate-gradient-descent",
            "12.3.3_adaptive-methods",
            "12.3.4_summary",
            "12.3.5_exercises"
        ],
        "12.4_stochastic-gradient-descent": [
            "12.4.1_stochastic-gradient-updates",
            "12.4.2_dynamic-learning-rate",
            "12.4.3_convergence-analysis-for-convex-objectives",
            "12.4.4_stochastic-gradients-and-finite-samples",
            "12.4.5_summary",
            "12.4.6_exercises"
        ],
        "12.5_minibatch-stochastic-gradient-descent": [
            "12.5.1_vectorization-and-caches",
            "12.5.2_minibatches",
            "12.5.3_reading-the-dataset",
            "12.5.4_implementation-from-scratch",
            "12.5.5_concise-implementation",
            "12.5.6_summary",
            "12.5.7_exercises"
        ],
        "12.6_momentum": [
            "12.6.1_basics",
            "12.6.2_practical-experiments",
            "12.6.3_theoretical-analysis",
            "12.6.4_summary",
            "12.6.5_exercises"
        ],
        "12.7_adagrad": [
            "12.7.1_sparse-features-and-learning-rates",
            "12.7.2_preconditioning",
            "12.7.3_the-algorithm",
            "12.7.4_implementation-from-scratch",
            "12.7.5_concise-implementation",
            "12.7.6_summary",
            "12.7.7_exercises"
        ],
        "12.8_rmsprop": [
            "12.8.1_the-algorithm",
            "12.8.2_implementation-from-scratch",
            "12.8.3_concise-implementation",
            "12.8.4_summary",
            "12.8.5_exercises"
        ],
        "12.9_adadelta": [
            "12.9.1_the-algorithm",
            "12.9.2_implementation",
            "12.9.3_summary",
            "12.9.4_exercises"
        ],
        "12.10_adam": [
            "12.10.1_the-algorithm",
            "12.10.2_implementation",
            "12.10.3_yogi",
            "12.10.4_summary",
            "12.10.5_exercises"
        ],
        "12.11_learning-rate-scheduling": [
            "12.11.1_toy-problem",
            "12.11.2_schedulers",
            "12.11.3_policies",
            "12.11.4_summary",
            "12.11.5_exercises"
        ]
    },
    "13_computational-performance": {
        "13.1_compilers-and-interpreters": [
            "13.1.1_symbolic-programming",
            "13.1.2_hybrid-programming",
            "13.1.3_hybridizing-the-sequential-class",
            "13.1.4_summary",
            "13.1.5_exercises"
        ],
        "13.2_asynchronous-computation": [
            "13.2.1_asynchrony-via-backend",
            "13.2.2_barriers-and-blockers",
            "13.2.3_improving-computation",
            "13.2.4_summary",
            "13.2.5_exercises"
        ],
        "13.3_automatic-parallelism": [
            "13.3.1_parallel-computation-on-gpus",
            "13.3.2_parallel-computation-and-communication",
            "13.3.3_summary",
            "13.3.4_exercises"
        ],
        "13.4_hardware": [
            "13.4.1_computers",
            "13.4.2_memory",
            "13.4.3_storage",
            "13.4.4_cpus",
            "13.4.5_gpus-and-other-accelerators",
            "13.4.6_networks-and-buses",
            "13.4.7_more-latency-numbers",
            "13.4.8_summary",
            "13.4.9_exercises"
        ],
        "13.5_training-on-multiple-gpus": [
            "13.5.1_splitting-the-problem",
            "13.5.2_data-parallelism",
            "13.5.3_a-toy-network",
            "13.5.4_data-synchronization",
            "13.5.5_distributing-data",
            "13.5.6_training",
            "13.5.7_summary",
            "13.5.8_exercises"
        ],
        "13.6_concise-implementation-for-multiple-gpus": [
            "13.6.1_a-toy-network",
            "13.6.2_network-initialization",
            "13.6.3_training",
            "13.6.4_summary",
            "13.6.5_exercises"
        ],
        "13.7_parameter-servers": [
            "13.7.1_data-parallel-training",
            "13.7.2_ring-synchronization",
            "13.7.3_multi-machine-training",
            "13.7.4_key-value-stores",
            "13.7.5_summary",
            "13.7.6_exercises"
        ]
    },
    "14_computer-vision": {
        "14.1_image-augmentation": [
            "14.1.1_common-image-augmentation-methods",
            "14.1.2_training-with-image-augmentation",
            "14.1.3_summary",
            "14.1.4_exercises"
        ],
        "14.2_fine-tuning": [
            "14.2.1_steps",
            "14.2.2_hot-dog-recognition",
            "14.2.3_summary",
            "14.2.4_exercises"
        ],
        "14.3_object-detection-and-bounding-boxes": [
            "14.3.1_bounding-boxes",
            "14.3.2_summary",
            "14.3.3_exercises"
        ],
        "14.4_anchor-boxes": [
            "14.4.1_generating-multiple-anchor-boxes",
            "14.4.2_intersection-over-union-iou",
            "14.4.3_labeling-anchor-boxes-in-training-data",
            "14.4.4_predicting-bounding-boxes-with-non-maximum-suppression",
            "14.4.5_summary",
            "14.4.6_exercises"
        ],
        "14.5_multiscale-object-detection": [
            "14.5.1_multiscale-anchor-boxes",
            "14.5.2_multiscale-detection",
            "14.5.3_summary",
            "14.5.4_exercises"
        ],
        "14.6_the-object-detection-dataset": [
            "14.6.1_downloading-the-dataset",
            "14.6.2_reading-the-dataset",
            "14.6.3_demonstration",
            "14.6.4_summary",
            "14.6.5_exercises"
        ],
        "14.7_single-shot-multibox-detection": [
            "14.7.1_model",
            "14.7.2_training",
            "14.7.3_prediction",
            "14.7.4_summary",
            "14.7.5_exercises"
        ],
        "14.8_region-based-cnns-r-cnns": [
            "14.8.1_r-cnns",
            "14.8.2_fast-r-cnn",
            "14.8.3_faster-r-cnn",
            "14.8.4_mask-r-cnn",
            "14.8.5_summary",
            "14.8.6_exercises"
        ],
        "14.9_semantic-segmentation-and-the-dataset": [
            "14.9.1_image-segmentation-and-instance-segmentation",
            "14.9.2_the-pascal-voc2012-semantic-segmentation-dataset",
            "14.9.3_summary",
            "14.9.4_exercises"
        ],
        "14.10_transposed-convolution": [
            "14.10.1_basic-operation",
            "14.10.2_padding-strides-and-multiple-channels",
            "14.10.3_connection-to-matrix-transposition",
            "14.10.4_summary",
            "14.10.5_exercises"
        ],
        "14.11_fully-convolutional-networks": [
            "14.11.1_the-model",
            "14.11.2_initializing-transposed-convolutional-layers",
            "14.11.3_reading-the-dataset",
            "14.11.4_training",
            "14.11.5_prediction",
            "14.11.6_summary",
            "14.11.7_exercises"
        ],
        "14.12_neural-style-transfer": [
            "14.12.1_method",
            "14.12.2_reading-the-content-and-style-images",
            "14.12.3_preprocessing-and-postprocessing",
            "14.12.4_extracting-features",
            "14.12.5_defining-the-loss-function",
            "14.12.6_initializing-the-synthesized-image",
            "14.12.7_training",
            "14.12.8_summary",
            "14.12.9_exercises"
        ],
        "14.13_image-classification-cifar-10-on-kaggle": [
            "14.13.1_obtaining-and-organizing-the-dataset",
            "14.13.2_image-augmentation",
            "14.13.3_reading-the-dataset",
            "14.13.4_defining-the-model",
            "14.13.5_defining-the-training-function",
            "14.13.6_training-and-validating-the-model",
            "14.13.7_classifying-the-testing-set-and-submitting-results-on-kaggle",
            "14.13.8_summary",
            "14.13.9_exercises"
        ],
        "14.14_dog-breed-identification-imagenet-dogs-on-kaggle": [
            "14.14.1_obtaining-and-organizing-the-dataset",
            "14.14.2_image-augmentation",
            "14.14.3_reading-the-dataset",
            "14.14.4_fine-tuning-a-pretrained-model",
            "14.14.5_defining-the-training-function",
            "14.14.6_training-and-validating-the-model",
            "14.14.7_classifying-the-testing-set-and-submitting-results-on-kaggle",
            "14.14.8_summary",
            "14.14.9_exercises"
        ]
    },
    "15_natural-language-processing-pretraining": {
        "15.1_word-embedding-word2vec": [
            "15.1.1_one-hot-vectors-are-a-bad-choice",
            "15.1.2_self-supervised-word2vec",
            "15.1.3_the-skip-gram-model",
            "15.1.4_the-continuous-bag-of-words-cbow-model",
            "15.1.5_summary",
            "15.1.6_exercises"
        ],
        "15.2_approximate-training": [
            "15.2.1_negative-sampling",
            "15.2.2_hierarchical-softmax",
            "15.2.3_summary",
            "15.2.4_exercises"
        ],
        "15.3_the-dataset-for-pretraining-word-embeddings": [
            "15.3.1_reading-the-dataset",
            "15.3.2_subsampling",
            "15.3.3_extracting-center-words-and-context-words",
            "15.3.4_negative-sampling",
            "15.3.5_loading-training-examples-in-minibatches",
            "15.3.6_putting-it-all-together",
            "15.3.7_summary",
            "15.3.8_exercises"
        ],
        "15.4_pretraining-word2vec": [
            "15.4.1_the-skip-gram-model",
            "15.4.2_training",
            "15.4.3_applying-word-embeddings",
            "15.4.4_summary",
            "15.4.5_exercises"
        ],
        "15.5_word-embedding-with-global-vectors-glove": [
            "15.5.1_skip-gram-with-global-corpus-statistics",
            "15.5.2_the-glove-model",
            "15.5.3_interpreting-glove-from-the-ratio-of-co-occurrence-probabilities",
            "15.5.4_summary",
            "15.5.5_exercises"
        ],
        "15.6_subword-embedding": [
            "15.6.1_the-fasttext-model",
            "15.6.2_byte-pair-encoding",
            "15.6.3_summary",
            "15.6.4_exercises"
        ],
        "15.7_word-similarity-and-analogy": [
            "15.7.1_loading-pretrained-word-vectors",
            "15.7.2_applying-pretrained-word-vectors",
            "15.7.3_summary",
            "15.7.4_exercises"
        ],
        "15.8_bidirectional-encoder-representations-from-transformers-bert": [
            "15.8.1_from-context-independent-to-context-sensitive",
            "15.8.2_from-task-specific-to-task-agnostic",
            "15.8.3_bert-combining-the-best-of-both-worlds",
            "15.8.4_input-representation",
            "15.8.5_pretraining-tasks",
            "15.8.6_putting-it-all-together",
            "15.8.7_summary",
            "15.8.8_exercises"
        ],
        "15.9_the-dataset-for-pretraining-bert": [
            "15.9.1_defining-helper-functions-for-pretraining-tasks",
            "15.9.2_transforming-text-into-the-pretraining-dataset",
            "15.9.3_summary",
            "15.9.4_exercises"
        ],
        "15.10_pretraining-bert": [
            "15.10.1_pretraining-bert",
            "15.10.2_representing-text-with-bert",
            "15.10.3_summary",
            "15.10.4_exercises"
        ]
    },
    "16_natural-language-processing-applications": {
        "16.1_sentiment-analysis-and-the-dataset": [
            "16.1.1_reading-the-dataset",
            "16.1.2_preprocessing-the-dataset",
            "16.1.3_creating-data-iterators",
            "16.1.4_putting-it-all-together",
            "16.1.5_summary",
            "16.1.6_exercises"
        ],
        "16.2_sentiment-analysis-using-recurrent-neural-networks": [
            "16.2.1_representing-single-text-with-rnns",
            "16.2.2_loading-pretrained-word-vectors",
            "16.2.3_training-and-evaluating-the-model",
            "16.2.4_summary",
            "16.2.5_exercises"
        ],
        "16.3_sentiment-analysis-using-convolutional-neural-networks": [
            "16.3.1_one-dimensional-convolutions",
            "16.3.2_max-over-time-pooling",
            "16.3.3_the-textcnn-model",
            "16.3.4_summary",
            "16.3.5_exercises"
        ],
        "16.4_natural-language-inference-and-the-dataset": [
            "16.4.1_natural-language-inference",
            "16.4.2_the-stanford-natural-language-inference-snli-dataset",
            "16.4.3_summary",
            "16.4.4_exercises"
        ],
        "16.5_natural-language-inference-using-attention": [
            "16.5.1_the-model",
            "16.5.2_training-and-evaluating-the-model",
            "16.5.3_summary",
            "16.5.4_exercises"
        ],
        "16.6_fine-tuning-bert-for-sequence-level-and-token-level-applications": [
            "16.6.1_single-text-classification",
            "16.6.2_text-pair-classification-or-regression",
            "16.6.3_text-tagging",
            "16.6.4_question-answering",
            "16.6.5_summary",
            "16.6.6_exercises"
        ],
        "16.7_natural-language-inference-fine-tuning-bert": [
            "16.7.1_loading-pretrained-bert",
            "16.7.2_the-dataset-for-fine-tuning-bert",
            "16.7.3_fine-tuning-bert",
            "16.7.4_summary",
            "16.7.5_exercises"
        ]
    },
    "17_reinforcement-learning": {
        "17.1_markov-decision-process-mdp": [
            "17.1.1_definition-of-an-mdp",
            "17.1.2_return-and-discount-factor",
            "17.1.3_discussion-of-the-markov-assumption",
            "17.1.4_summary",
            "17.1.5_exercises"
        ],
        "17.2_value-iteration": [
            "17.2.1_stochastic-policy",
            "17.2.2_value-function",
            "17.2.3_action-value-function",
            "17.2.4_optimal-stochastic-policy",
            "17.2.5_principle-of-dynamic-programming",
            "17.2.6_value-iteration",
            "17.2.7_policy-evaluation",
            "17.2.8_implementation-of-value-iteration",
            "17.2.9_summary",
            "17.2.10_exercises"
        ],
        "17.3_q-learning": [
            "17.3.1_the-q-learning-algorithm",
            "17.3.2_an-optimization-problem-underlying-q-learning",
            "17.3.3_exploration-in-q-learning",
            "17.3.4_the-self-correcting-property-of-q-learning",
            "17.3.5_implementation-of-q-learning",
            "17.3.6_summary",
            "17.3.7_exercises"
        ]
    },
    "18_gaussian-processes": {
        "18.1_introduction-to-gaussian-processes": [
            "18.1.1_summary",
            "18.1.2_exercises"
        ],
        "18.2_gaussian-process-priors": [
            "18.2.1_definition",
            "18.2.2_a-simple-gaussian-process",
            "18.2.3_from-weight-space-to-function-space",
            "18.2.4_the-radial-basis-function-rbf-kernel",
            "18.2.5_the-neural-network-kernel",
            "18.2.6_summary",
            "18.2.7_exercises"
        ],
        "18.3_gaussian-process-inference": [
            "18.3.1_posterior-inference-for-regression",
            "18.3.2_equations-for-making-predictions-and-learning-kernel-hyperparameters-in-gp-regression",
            "18.3.3_interpreting-equations-for-learning-and-predictions",
            "18.3.4_worked-example-from-scratch",
            "18.3.5_making-life-easy-with-gpytorch",
            "18.3.6_summary",
            "18.3.7_exercises"
        ]
    },
    "19_hyperparameter-optimization": {
        "19.1_what-is-hyperparameter-optimization": [
            "19.1.1_the-optimization-problem",
            "19.1.2_random-search",
            "19.1.3_summary",
            "19.1.4_exercises"
        ],
        "19.2_hyperparameter-optimization-api": [
            "19.2.1_searcher",
            "19.2.2_scheduler",
            "19.2.3_tuner",
            "19.2.4_bookkeeping-the-performance-of-hpo-algorithms",
            "19.2.5_example-optimizing-the-hyperparameters-of-a-convolutional-neural-network",
            "19.2.6_comparing-hpo-algorithms",
            "19.2.7_summary",
            "19.2.8_exercises"
        ],
        "19.3_asynchronous-random-search": [
            "19.3.1_objective-function",
            "19.3.2_asynchronous-scheduler",
            "19.3.3_visualize-the-asynchronous-optimization-process",
            "19.3.4_summary",
            "19.3.5_exercises"
        ],
        "19.4_multi-fidelity-hyperparameter-optimization": [
            "19.4.1_successive-halving",
            "19.4.2_summary"
        ],
        "19.5_asynchronous-successive-halving": [
            "19.5.1_objective-function",
            "19.5.2_asynchronous-scheduler",
            "19.5.3_visualize-the-optimization-process",
            "19.5.4_summary"
        ]
    },
    "20_generative-adversarial-networks": {
        "20.1_generative-adversarial-networks": [
            "20.1.1_generate-some-real-data",
            "20.1.2_generator",
            "20.1.3_discriminator",
            "20.1.4_training",
            "20.1.5_summary",
            "20.1.6_exercises"
        ],
        "20.2_deep-convolutional-generative-adversarial-networks": [
            "20.2.1_the-pokemon-dataset",
            "20.2.2_the-generator",
            "20.2.3_discriminator",
            "20.2.4_training",
            "20.2.5_summary",
            "20.2.6_exercises"
        ]
    },
    "21_recomender-systems": {
        "21.1_overview-of-recommender-systems": [
            "21.1.1_collaborative-filtering",
            "21.1.2_explicit-feedback-and-implicit-feedback",
            "21.1.3_recommendation-tasks",
            "21.1.4_summary",
            "21.1.5_exercises"
        ]
    }
}

# Boş bir Jupyter notebook şablonu
notebook_template = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# {}"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def create_notebook(path, title):
    """Boş bir Jupyter notebook oluştur"""
    notebook = notebook_template.copy()
    notebook["cells"][0]["source"] = [f"# {title}"]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

def create_structure():
    """Tüm klasör yapısını ve notebook dosyalarını oluştur"""
    base_path = "notebooks"

    for chapter_folder, subsections in structure.items():
        chapter_path = os.path.join(base_path, chapter_folder)

        # Ana klasör yoksa oluştur
        if not os.path.exists(chapter_path):
            os.makedirs(chapter_path)
            print(f"Created: {chapter_path}")

        for subsection_name, subsubsections in subsections.items():
            subsection_path = os.path.join(chapter_path, subsection_name)

            # Alt başlık klasörünü oluştur
            if not os.path.exists(subsection_path):
                os.makedirs(subsection_path)
                print(f"Created: {subsection_path}")

            # Eğer alt-alt başlıklar varsa, her biri için notebook oluştur
            if subsubsections:
                for subsubsection_name in subsubsections:
                    notebook_filename = f"{subsubsection_name}.ipynb"
                    notebook_path = os.path.join(subsection_path, notebook_filename)

                    # Notebook yoksa oluştur
                    if not os.path.exists(notebook_path):
                        # Başlığı düzenle (tire ve alt çizgiyi boşluğa çevir, her kelimenin ilk harfini büyüt)
                        title = subsubsection_name.replace('-', ' ').replace('_', ' ').title()
                        create_notebook(notebook_path, title)
                        print(f"Created: {notebook_path}")
            else:
                # Alt-alt başlık yoksa, ana başlık için bir notebook oluştur
                notebook_filename = f"{subsection_name}.ipynb"
                notebook_path = os.path.join(subsection_path, notebook_filename)

                if not os.path.exists(notebook_path):
                    title = subsection_name.replace('-', ' ').replace('_', ' ').title()
                    create_notebook(notebook_path, title)
                    print(f"Created: {notebook_path}")

if __name__ == "__main__":
    print("Creating D2L structure...")
    create_structure()
    print("Done!")
