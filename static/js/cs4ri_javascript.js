
Blockly.JavaScript['model_type'] = function(block) {
    return [block.getFieldValue('MODEL_TYPE'), Blockly.JavaScript.ORDER_FUNCTION_CALL];
}

Blockly.JavaScript['run_model'] = function(block) {
    var model = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC);
    var inputs = Blockly.JavaScript.valueToCode(block, 'INPUTS', Blockly.JavaScript.ORDER_ATOMIC);
    var code = "model="+model; 
    var train = {"inputs": inputs, "model": model};
    console.log(train);
    return [code, Blockly.JavaScript.ORDER_FUNCTION_CALL];
};
