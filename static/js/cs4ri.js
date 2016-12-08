

// TODO: get array of inputs via service
var model_inputs = [["predicted tide", "TIDE"], ["predicted wind speed", "WIND_SPEED"], ["predicted barometric pressure", "PRESSURE"], ["measured water level", "ZETA"]];
Blockly.Blocks['model_input_type'] = {
    init: function() {
        this.appendDummyInput().appendField("input type").appendField(new Blockly.FieldDropdown(model_inputs), "MODEL_INPUT_TYPE");
        this.setOutput(true, null);
        this.setColour(170); // GREEN == INPUT
        this.setTooltip('select an input type');
    }
};

var model_types = [["Simple", "REGRESSION"], ["Complex #1", "NEURAL1"], ["Complex #2", "NEURAL2"], ["Surprise", "FOREST"]];
Blockly.Blocks['model_type'] = {
    init: function() {
        this.appendDummyInput().appendField("model type").appendField(new Blockly.FieldDropdown(model_types), "MODEL_TYPE");
        this.setOutput(true, null);
        this.setColour(120); // GREEN == INPUT
        this.setTooltip('select a model type');
    }
};

Blockly.Blocks['run_model'] = {
    init: function() {
        this.appendValueInput("MODEL").setCheck(null).appendField("model");
        this.appendValueInput("INPUTS").setCheck("Array").appendField("inputs");
        this.setPreviousStatement(true, null);
        //this.setColour(330); // BLACK == black-box
        this.setTooltip('model to run');
    }
};
/*
Blockly.JavaScript['run_model'] = function(block) {
    var value_model = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC);
    var value_inputs = Blockly.JavaScript.valueToCode(block, 'INPUTS', Blockly.JavaScript.ORDER_ATOMIC);
    // TODO: Assemble JavaScript into code variable.
    var code = '...;\n';
    return code;
};
*/




var number_of_model_inputs = [["1","1"],["2","2"],["3","3"],["4","4"],["5","5"],["6","6"],["7","7"],["8","8"],["9","9"],["10","10"]];
Blockly.Blocks['model_inputs'] = {
    init: function() {
        this.appendDummyInput().appendField("number of inputs").appendField(new Blockly.FieldDropdown(number_of_model_inputs), "NUMBER_OF_MODEL_INPUTS");
        this.appendValueInput("model_input_1").appendField("input #1").setCheck(null);
        this.setOutput(true, null);
        this.setColour(170); // GREEN == INPUT
        this.setTooltip('');
    },
    onchange: function(event) {
        if( event.type == Blockly.Events.CHANGE && event.blockId == this.id ) {
            number_of_inputs = this.getField("NUMBER_OF_MODEL_INPUTS").getValue();
            number_of_connections = this.inputList.length-1;
            if( number_of_inputs > number_of_connections ) {
                for( i=number_of_connections; i<number_of_inputs; i++ ) { 
                    this.appendValueInput("model_input_"+(i+1)).appendField("input #"+(i+1)).setCheck(null);
                }
            }
            else if( number_of_inputs < number_of_connections ) {
                for( i=number_of_connections; i>number_of_inputs; i-- ) { 
                    this.removeInput("model_input_"+i);
                }
            }
        }
    }
};




Blockly.Blocks['print_error'] = {
    init: function() {
        this.appendDummyInput().appendField("error_undefined_item");
        this.setPreviousStatement(true, null);
        this.setColour(0); // RED == ERROR
        this.setTooltip('');
        this.setHelpUrl('http://www.example.com/');
    }
};



/* ---- */
/* main */
/* ---- */
var workspace = Blockly.inject('blocklyDiv', {
    media: 'media/',
    toolbox: document.getElementById('toolbox')
});

//var code = Blockly.Python.workspaceToCode(workspace);
//function updateFunction(event) {
//    code = Blockly.Python.workspaceToCode(workspace);
//}
//workspace.addChangeListener(updateFunction);

$.get({
    url: '/blocks',
    success: function(xml_text) {
        var xml = Blockly.Xml.textToDom(xml_text);
        Blockly.Xml.domToWorkspace(xml, workspace);
    }
});

function saveBlock() {
    var xml = Blockly.Xml.workspaceToDom(workspace);
    var xml_text = Blockly.Xml.domToText(xml);
    console.log(xml_text);
    $.post({
        url: '/blocks',
        data: {"xml": xml_text},
    });    
}
