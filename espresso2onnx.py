import os
import argparse
import struct
import json
import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np


def to_float16(bin: bytes):
    return np.frombuffer(bin, dtype=np.float16)


def to_float(bin: bytes):
    return np.frombuffer(bin, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coreml_folder', help='Input coreml model folder')
    args = parser.parse_args()
    coreml_folder = args.coreml_folder

    layer_bytes = []
    net_layer_data = {}
    coreml_net = os.path.join(coreml_folder, 'model.espresso.net')
    coreml_shape = os.path.join(coreml_folder, 'model.espresso.shape')
    coreml_weights = os.path.join(coreml_folder, 'model.espresso.weights')

    with open(coreml_net, encoding='utf-8') as f:
        net_dict = json.load(f)
        net_layers = net_dict['layers']

    with open(coreml_shape, encoding='utf-8') as f:
        net_dict = json.load(f)
        net_layer_shapes = net_dict['layer_shapes']

    with open(coreml_weights, 'rb') as f:
        num_layers = struct.unpack('<i', f.read(4))[0]
        f.read(4)  # padding bytes

        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))

        for layer_num, num_bytes in layer_bytes:
            raw_data = f.read(num_bytes)
            net_layer_data[layer_num] = raw_data

    net_inputes = []
    net_input_names = net_layers[0]['bottom'].split(',')
    print(net_input_names, net_layers)
    for net_input_name in net_input_names:
        net_input_shape_dict = net_layer_shapes[net_input_name]
        net_input = helper.make_tensor_value_info(net_input_name, TensorProto.FLOAT,
                                                  [net_input_shape_dict['n'], net_input_shape_dict['k'],
                                                   net_input_shape_dict['h'], net_input_shape_dict['w']])
        net_inputes.append(net_input)

    net_output_shape_dict = net_layer_shapes[net_layers[-1]['top']]
    net_output = helper.make_tensor_value_info(net_layers[-1]['top'], TensorProto.FLOAT,
                                               [net_output_shape_dict['n'], net_output_shape_dict['k'],
                                                net_output_shape_dict['h'], net_output_shape_dict['w']])
    net_outputes = [net_output]

    net_output_name = net_layers[-1]['top']
    if not net_output_name.isdigit():
        net_output2_shape_dict = net_layer_shapes[net_output_name]
        net_output2 = helper.make_tensor_value_info(net_output_name, TensorProto.FLOAT,
                                                    [net_output2_shape_dict['n'], net_output2_shape_dict['k'],
                                                     net_output2_shape_dict['h'], net_output2_shape_dict['w']])
        net_outputes.append(net_output2)

    onnx_blob_shapes = []
    for blob_name, blob_shape_dict in net_layer_shapes.items():
        onnx_blob_shapes.append(helper.make_tensor_value_info(blob_name, TensorProto.FLOAT,
                                                              [blob_shape_dict['n'],
                                                               blob_shape_dict['k'],
                                                               blob_shape_dict['h'],
                                                               blob_shape_dict['w']]))

    onnx_net_nodes = []
    onnx_net_weights = []

    produced_tensors = set(net_input_names)

    for layer_info in net_layers:
        print(layer_info['type'])
        if layer_info['type'] == 'convolution':
            stride_x = layer_info.get('stride_x', 1)
            stride_y = layer_info.get('stride_y', 1)
            auto_pad = 'SAME_UPPER' if layer_info['pad_mode'] == 1 else None

            node_inputs = layer_info['bottom'].split(',')
            node_inputs.append(str(layer_info.get('blob_weights', layer_info.get('blob_weights_f16'))))

            if 'blob_biases' in layer_info:
                node_inputs.append(str(layer_info['blob_biases']))

            node_conv_outputs = layer_info['top'].split(',')
            node_relu_outputs = []

            if layer_info['fused_relu'] == 1:
                node_relu_outputs = node_conv_outputs
                node_conv_outputs = ['conv_' + output for output in node_relu_outputs]
                for temp_output in node_relu_outputs:
                    blob_shape_dict = net_layer_shapes[temp_output]
                    onnx_blob_shapes.append(helper.make_tensor_value_info('conv_' + temp_output, TensorProto.FLOAT,
                                                                          [blob_shape_dict['n'],
                                                                           blob_shape_dict['k'],
                                                                           blob_shape_dict['h'],
                                                                           blob_shape_dict['w']]))

            conv_group_num = layer_info['n_groups']
            conv_node = helper.make_node('Conv',  # node type
                                         node_inputs,  # inputs
                                         node_conv_outputs,  # outputs
                                         kernel_shape=[layer_info['Nx'], layer_info['Ny']],
                                         strides=[stride_x, stride_y],
                                         auto_pad=auto_pad,
                                         group=conv_group_num,
                                         dilations=[1, 1],
                                         pads=[layer_info.get('pad_l', 0), layer_info.get('pad_t', 0),
                                               layer_info.get('pad_r', 0), layer_info.get('pad_b', 0)] if not auto_pad else None)
            onnx_net_nodes.append(conv_node)

            weights_key = str(layer_info.get('blob_weights_f16', layer_info.get('blob_weights')))
            weights_shape = [layer_info['C'], int(layer_info['K'] / conv_group_num), layer_info['Nx'], layer_info['Ny']]
            weights_tensor = helper.make_tensor(weights_key, TensorProto.FLOAT, weights_shape,
                                                tuple(to_float16(net_layer_data[int(weights_key)]).astype(np.float32) if 'blob_weights_f16' in layer_info else to_float(net_layer_data[int(weights_key)])))
            onnx_net_weights.append(weights_tensor)

            if 'blob_biases' in layer_info:
                bias_shape = [layer_info['C']]
                bias_tensor = helper.make_tensor(str(layer_info['blob_biases']), TensorProto.FLOAT, bias_shape,
                                                 tuple(to_float(net_layer_data[layer_info['blob_biases']])))
                onnx_net_weights.append(bias_tensor)

            if layer_info['fused_relu'] == 1:
                relu_node = helper.make_node('Relu',  # node type
                                             node_conv_outputs,  # inputs
                                             node_relu_outputs)  # outputs
                onnx_net_nodes.append(relu_node)

            produced_tensors.update(node_conv_outputs)
            produced_tensors.update(node_relu_outputs)

        elif layer_info['type'] == 'pool':
            stride_x = layer_info.get('stride_x', 1)
            stride_y = layer_info.get('stride_y', 1)
            node_type = 'AveragePool' if layer_info['avg_or_max'] == 0 else 'MaxPool'
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            pool_node = helper.make_node(node_type,  # node type
                                         node_inputs,  # inputs
                                         node_outputs,  # outputs
                                         kernel_shape=[layer_info['size_x'], layer_info['size_y']],
                                         strides=[stride_x, stride_y],
                                         pads=[layer_info.get('pad_l', 0), layer_info.get('pad_t', 0),
                                               layer_info.get('pad_r', 0), layer_info.get('pad_b', 0)])
            onnx_net_nodes.append(pool_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'pool3d':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            pooling_mode = 'AveragePool' if layer_info.get('avg_or_max', 0) == 0 else 'MaxPool'
            pool_node = helper.make_node(
                pooling_mode,
                node_inputs,
                node_outputs,
                kernel_shape=[layer_info['size_t'], layer_info['size_y'], layer_info['size_x']],
                strides=[layer_info['stride_t'], layer_info['stride_y'], layer_info['stride_x']],
                pads=[layer_info.get('pad_front', 0), layer_info.get('pad_t', 0), layer_info.get('pad_l', 0),
                      layer_info.get('pad_back', 0), layer_info.get('pad_b', 0), layer_info.get('pad_r', 0)]
            )
            onnx_net_nodes.append(pool_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'elementwise':
            node_inputs = layer_info['bottom'].split(',')
            node_type = ''
            node_inputs_extra = []

            if layer_info['operation'] in [0, 2]:  # Add
                node_type = 'Add'
                if len(node_inputs) == 1:
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(scales_tensor)
            elif layer_info['operation'] == 1:  # Mul
                node_type = 'Mul'
                if len(node_inputs) == 1:
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(scales_tensor)
            elif layer_info['operation'] == -999:  # Sub
                node_type = 'Sub'
                if len(node_inputs) == 1:
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(scales_tensor)
            elif layer_info['operation'] == 3:  # Mul
                node_type = 'Mul'
                if len(node_inputs) == 1:
                    scales_tensor_name = 'elementwise_' + layer_info['top']
                    node_inputs_extra.append(scales_tensor_name)
                    scales = [layer_info['alpha']]
                    scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                    onnx_net_weights.append(scales_tensor)
            elif layer_info['operation'] == 10:  # Div
                node_type = 'Div'
                scales_tensor_name = 'elementwise_' + layer_info['top']
                node_inputs_extra.append(scales_tensor_name)
                scales = [layer_info['alpha']]
                scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [1], scales)
                onnx_net_weights.append(scales_tensor)
            elif layer_info['operation'] == 24:  # Abs
                node_type = 'Abs'
            elif layer_info['operation'] == 105:  # Custom operation mapping if needed
                # Add your custom operation mapping here if necessary
                pass
            elif layer_info['operation'] == 106:  # Custom operation mapping if needed
                node_type = 'Greater'
            elif layer_info['operation'] == 119:  # Clip
                node_type = 'Clip'
                alpha_tensor_name = 'elementwise_' + layer_info['top'] + 'alpha'
                beta_tensor_name = 'elementwise_' + layer_info['top'] + 'beta'
                node_inputs_extra.append(alpha_tensor_name)
                node_inputs_extra.append(beta_tensor_name)
                alpha = [layer_info.get('alpha', 0.0)]
                beta = [layer_info.get('beta', 1.0)]
                alpha_tensor = helper.make_tensor(alpha_tensor_name, TensorProto.FLOAT, [1], alpha)
                beta_tensor = helper.make_tensor(beta_tensor_name, TensorProto.FLOAT, [1], beta)
                onnx_net_weights.append(alpha_tensor)
                onnx_net_weights.append(beta_tensor)
            else:
                print('Error: unsupported elementwise operation: ' + str(layer_info['operation']))
                assert False

            node_inputs.extend(node_inputs_extra)
            node_outputs = layer_info['top'].split(',')
            elementwise_node = helper.make_node(node_type,  # node type
                                                node_inputs,  # inputs
                                                node_outputs)  # outputs
            onnx_net_nodes.append(elementwise_node)


        elif layer_info['type'] == 'upsample':
            node_type = 'Upsample'
            if layer_info['mode'] != 0:
                print('Error: unsupported upsample mode: ' + str(layer_info['mode']))
                assert False

            scales_tensor_name = 'upsample_' + layer_info['top']
            node_inputs = layer_info['bottom'].split(',')
            node_inputs.append(scales_tensor_name)
            node_outputs = layer_info['top'].split(',')
            upsample_node = helper.make_node(node_type,  # node type
                                             node_inputs,  # inputs
                                             node_outputs,  # outputs
                                             mode='nearest')
            onnx_net_nodes.append(upsample_node)

            scales = [1.0, 1.0, layer_info['scaling_factor_x'], layer_info['scaling_factor_y']]
            scales_tensor = helper.make_tensor(scales_tensor_name, TensorProto.FLOAT, [4], scales)
            onnx_net_weights.append(scales_tensor)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'concat':
            node_type = 'Concat'
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            concat_node = helper.make_node(node_type,  # node type
                                           node_inputs,  # inputs
                                           node_outputs,  # outputs
                                           axis=1)
            onnx_net_nodes.append(concat_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'activation':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            activation_mode = layer_info['mode']

            activation_map = {
                0: 'Relu',
                1: 'Tanh',
                2: 'LeakyRelu',
                3: 'Sigmoid',
                4: 'PRelu',
                8: 'Elu',
                9: 'ThresholdedRelu',
                10: 'Softplus',
                12: 'Softsign'
            }

            if activation_mode in activation_map:
                layer_node = helper.make_node(activation_map[activation_mode],  # node type
                                              node_inputs,  # inputs
                                              node_outputs,  # outputs,
                                              alpha=layer_info.get('alpha', 0.0) if activation_mode in [2, 9] else None)
                onnx_net_nodes.append(layer_node)
                produced_tensors.update(node_outputs)
            else:
                print('Error: unsupported activation mode: ' + str(activation_mode))
                assert False

        elif layer_info['type'] == 'batchnorm':
            node_inputs = layer_info['bottom'].split(',')
            weights_prefix = str(layer_info['blob_batchnorm_params'])
            node_inputs.append(weights_prefix + 's')
            node_inputs.append(weights_prefix + 'bias')

            channels = layer_info['C']
            node_bn_outputs = layer_info['top'].split(',')
            data = to_float(net_layer_data[layer_info['blob_batchnorm_params']])
            data = data.reshape([channels, 4])
            s = data[:, 0]
            bias = data[:, 1]

            layer_node = helper.make_node('InstanceNormalization',  # node type
                                          node_inputs,  # inputs
                                          node_bn_outputs,  # outputs
                                          epsilon=layer_info['training_eps'])
            onnx_net_nodes.append(layer_node)
            produced_tensors.update(node_bn_outputs)

            weights_shape = [layer_info['C']]
            s_tensor = helper.make_tensor(weights_prefix + 's', TensorProto.FLOAT, weights_shape, s)
            bias_tensor = helper.make_tensor(weights_prefix + 'bias', TensorProto.FLOAT, weights_shape, bias)
            onnx_net_weights.append(s_tensor)
            onnx_net_weights.append(bias_tensor)

        elif layer_info['type'] == 'inner_product':
            stride_x = layer_info.get('stride_x', 1)
            stride_y = layer_info.get('stride_y', 1)
            node_inputs = layer_info['bottom'].split(',')
            node_inputs.append(str(layer_info.get('blob_weights', layer_info.get('blob_weights_f16'))))

            if 'blob_biases' in layer_info:
                node_inputs.append(str(layer_info['blob_biases']))

            node_conv_outputs = layer_info['top'].split(',')
            node_relu_outputs = []

            if layer_info['has_relu'] == 1:
                node_relu_outputs = node_conv_outputs
                node_conv_outputs = ['conv_' + output for output in node_relu_outputs]
                for temp_output in node_relu_outputs:
                    blob_shape_dict = net_layer_shapes[temp_output]
                    onnx_blob_shapes.append(helper.make_tensor_value_info('conv_' + temp_output, TensorProto.FLOAT,
                                                                          [blob_shape_dict['n'],
                                                                           blob_shape_dict['k'],
                                                                           blob_shape_dict['h'],
                                                                           blob_shape_dict['w']]))

            conv_node = helper.make_node('Conv',  # node type
                                         node_inputs,  # inputs
                                         node_conv_outputs,  # outputs
                                         kernel_shape=[layer_info.get('Nx', 1), layer_info.get('Ny', 1)],
                                         strides=[stride_x, stride_y],
                                         pads=[0, 0, 0, 0],
                                         group=1,
                                         dilations=[1, 1])
            onnx_net_nodes.append(conv_node)
            produced_tensors.update(node_conv_outputs)

            weights_shape = [layer_info['nC'], int(layer_info['nB']), layer_info.get('Nx', 1), layer_info.get('Ny', 1)]
            weights_tensor = helper.make_tensor(str(layer_info['blob_weights_f16']), TensorProto.FLOAT, weights_shape,
                                                tuple(to_float16(net_layer_data[layer_info['blob_weights_f16']]).astype(np.float32)))
            onnx_net_weights.append(weights_tensor)

            if 'blob_biases' in layer_info:
                bias_shape = [layer_info['nC']]
                bias_tensor = helper.make_tensor(str(layer_info['blob_biases']), TensorProto.FLOAT, bias_shape,
                                                 tuple(to_float(net_layer_data[layer_info['blob_biases']])))
                onnx_net_weights.append(bias_tensor)

            if layer_info['has_relu'] == 1:
                relu_node = helper.make_node('Relu',  # node type
                                             node_conv_outputs,  # inputs
                                             node_relu_outputs)  # outputs
                onnx_net_nodes.append(relu_node)
                produced_tensors.update(node_relu_outputs)

        elif layer_info['type'] == 'softmax':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            softmax_node = helper.make_node('Softmax',  # node type
                                            node_inputs,  # inputs
                                            node_outputs,  # outputs
                                            axis=1)
            onnx_net_nodes.append(softmax_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'expand_dims':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            axes = [layer_info['axes_0'], layer_info['axes_1']] if 'axes_0' in layer_info and 'axes_1' in layer_info else [layer_info['axes_0']]
            expand_node = helper.make_node('Unsqueeze',  # node type in ONNX
                                           node_inputs,  # inputs
                                           node_outputs,  # outputs
                                           axes=axes)
            onnx_net_nodes.append(expand_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'squeeze':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            if 'axes_0' in layer_info and 'axes_1' in layer_info:
                axes = [layer_info['axes_0'], layer_info['axes_1']]
                squeeze_node = helper.make_node('Squeeze',  # node type in ONNX
                                                node_inputs,  # inputs
                                                node_outputs,  # outputs
                                                axes=axes)
            else:
                squeeze_node = helper.make_node('Squeeze',  # node type in ONNX
                                                node_inputs,  # inputs
                                                node_outputs)  # outputs
            onnx_net_nodes.append(squeeze_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'crop':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            crop_node = helper.make_node('Crop',  # node type in ONNX
                                         node_inputs,  # inputs
                                         node_outputs,  # outputs,
                                         border=[layer_info['t'], layer_info['l'], layer_info['b'], layer_info['r']])
            onnx_net_nodes.append(crop_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'general_concat':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            concat_node = helper.make_node('Concat',  # node type in ONNX
                                           node_inputs,  # inputs
                                           node_outputs,  # outputs,
                                           axis=layer_info['axis'])
            onnx_net_nodes.append(concat_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'reshape':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            shape_tensor_name = "shape_tensor_" + layer_info['name']  # Ensure unique tensor name
            shape_tensor = helper.make_tensor(shape_tensor_name, TensorProto.INT64, [4],
                                            [layer_info['dst_n'], layer_info['dst_k'], layer_info['dst_h'], layer_info['dst_w']])
            onnx_net_weights.append(shape_tensor)
            reshape_node = helper.make_node('Reshape',  # node type in ONNX
                                            node_inputs + [shape_tensor_name],  # inputs
                                            node_outputs)  # outputs
            onnx_net_nodes.append(reshape_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'transpose':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            perm = [layer_info.get('axis_{}'.format(i), i) for i in range(len(node_inputs))]
            transpose_node = helper.make_node('Transpose',  # node type in ONNX
                                              node_inputs,  # inputs
                                              node_outputs,  # outputs
                                              perm=perm)  # permutation of axes
            onnx_net_nodes.append(transpose_node)
            produced_tensors.update(node_outputs)

        elif layer_info['type'] == 'copy':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            copy_node = helper.make_node('Identity',  # 'Copy' in ONNX is implemented using 'Identity'
                                        node_inputs,  # inputs
                                        node_outputs)  # outputs
            onnx_net_nodes.append(copy_node)
            
        elif layer_info['type'] == 'reduce':
            node_inputs = layer_info['bottom'].split(',')
            node_outputs = layer_info['top'].split(',')
            reduce_mode = layer_info.get('mode', 0)  # Default mode to 0 if not specified

            reduce_map = {
                0: 'ReduceSum',
                1: 'ReduceMean',
                2: 'ReduceProd',
                3: 'ReduceMin',
                4: 'ReduceMax',
                5: 'ReduceSumSquare',
                6: 'ReduceL1',
                7: 'ReduceL2',
                8: 'ReduceLogSum',
                9: 'ReduceLogSumExp'
            }

            if reduce_mode in reduce_map:
                reduce_node = helper.make_node(reduce_map[reduce_mode],  # node type
                                            node_inputs,  # inputs
                                            node_outputs,  # outputs
                                            keepdims=1)  # keeping the reduced dimensions
                onnx_net_nodes.append(reduce_node)
            else:
                print(f"Error: unsupported reduce mode: {reduce_mode}")
                assert False



        else:
            print('Error: unsupported layer type: ' + layer_info['type'])
            assert False

    graph_def = helper.make_graph(
        onnx_net_nodes,
        'onnx-model',
        net_inputes,
        net_outputes,
        initializer=onnx_net_weights,
        value_info=onnx_blob_shapes,
    )

    onnx_model = helper.make_model(graph_def, producer_name='YouTu Tencent',
                                   opset_imports=[helper.make_operatorsetid("", 12)])
    onnx_model.ir_version = 7

    onnx.checker.check_model(onnx_model)
    print('Before shape inference, the shape info of Y is:\n{}'.format(onnx_model.graph.value_info))

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))
    onnx.save(inferred_model, os.path.join(coreml_folder, 'model.onnx'))


if __name__ == '__main__':
    main()
