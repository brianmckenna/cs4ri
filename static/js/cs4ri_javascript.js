
Blockly.JavaScript['model_inputs'] = function(block) {
    var n = block.getFieldValue('NUMBER_OF_MODEL_INPUTS')
    var model_inputs = [];
    for(i = 1; i <= n; i++ ) {
        model_inputs.push(Blockly.JavaScript.valueToCode(block, 'MODEL_INPUT_'+i, Blockly.JavaScript.ORDER_ATOMIC));
    }
    var code = '[' + model_inputs.join() + ']';
    return [code, Blockly.JavaScript.ORDER_FUNCTION_CALL];
}


Blockly.JavaScript['model_input_type'] = function(block) {
    return [block.getFieldValue('MODEL_INPUT_TYPE'), Blockly.JavaScript.ORDER_FUNCTION_CALL];
}

Blockly.JavaScript['model_type'] = function(block) {
    return [block.getFieldValue('MODEL_TYPE'), Blockly.JavaScript.ORDER_FUNCTION_CALL];
}


Blockly.JavaScript['run_model'] = function(block) {
    var model = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC);
    var inputs = Blockly.JavaScript.valueToCode(block, 'INPUTS', Blockly.JavaScript.ORDER_ATOMIC);
    var code = 'var data = {"inputs": ' + inputs + ', "model": ' + model + '}';
    return [code, Blockly.JavaScript.ORDER_FUNCTION_CALL];
};
