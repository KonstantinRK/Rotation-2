import tensorflow as tf


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    for op in import_graph.get_operations():
        print(op)
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_graph(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        image_shape_v1 = [224, 224, 3]
        image_value_range = (-117, 255 - 117)
        endpoints_v1 = dict(
            inputs='input:0',
            outputs='output2:0'
        )
    frozen_func = wrap_frozen_graph(graph_def=graph_def, **endpoints_v1, print_graph=True)


def frozen_keras_graph(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.resource
    ]
    output_tensors = frozen_func.outputs

    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph)

    return graph_def


def write_graph(model):
    tf.io.write_graph(frozen_keras_graph(model), '.', 'frozen_graph.pb')


# model = tf.keras.applications.MobileNetV2()
# graph_def = frozen_keras_graph(model)
