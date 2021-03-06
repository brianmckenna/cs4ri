
/* ---------------- */
/* Number of Inputs */
/* ---------------- */
//var number_of_model_inputs = [["1","1"],["2","2"],["3","3"],["4","4"],["5","5"],["6","6"],["7","7"],["8","8"],["9","9"],["10","10"]];
var number_of_model_inputs = [["1","1"],["2","2"],["3","3"],["4","4"],["5","5"],["6","6"]];
Blockly.Blocks['model_inputs'] = {
    init: function() {
        this.appendDummyInput().appendField("number of inputs").appendField(new Blockly.FieldDropdown(number_of_model_inputs), "NUMBER_OF_MODEL_INPUTS");
        this.appendValueInput("MODEL_INPUT_1").appendField("input #1").setCheck(null);
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
                    this.appendValueInput("MODEL_INPUT_"+(i+1)).appendField("input #"+(i+1)).setCheck(null);
                }
            }
            else if( number_of_inputs < number_of_connections ) {
                for( i=number_of_connections; i>number_of_inputs; i-- ) {
                    this.removeInput("MODEL_INPUT_"+i);
                }
            }
        }
    }
};


/* ------------------------------------ */
/* Input Types                          */
/* ------------------------------------ */
var model_inputs = [["predicted tide", "k009"], ["24 predicted wind speed", "k243"], ["24 predicted barometric pressure", "k241"], ["24 measured water level", "k244"]];
$.get(
    '/inputs',
    function(res) {
        console.log(res);
        eval(res);
    }
);
Blockly.Blocks['model_input_type'] = {
    init: function() {
        this.appendDummyInput().appendField("input type").appendField(new Blockly.FieldDropdown(model_inputs), "MODEL_INPUT_TYPE");
        this.setOutput(true, null);
        this.setColour(170); // GREEN == INPUT
        this.setTooltip('select an input type');
    }
};

/* ------------------------------------ */
/* Model Types                          */
/* 0 Simple:     Linear Regression      */
/* 1 Complex #1: Neural Network         */
/* 2 Complex #2: Support Vector Machine */
/* 3 Complex #3: Random Forest          */
/* 4 Surprise:   Select a random one    */
/* ------------------------------------ */
var model_types = [["Simple: Linear Regression", "0"], ["Complex #1: Artificial Neural Network", "1"], ["Complex #2: Support Vector Machine", "2"], ["Complex #3: Random Forest Regression", "3"], ["Surprise Me!", "4"]];
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
    toolbox: document.getElementById('toolbox'),
    scrollbars: false,
    zoom: {
        controls: true,
        wheel: false,
        startScale: 0.7,
        maxScale: 1,
        minScale: 0.5,
        scaleSpeed: 0.5
    },
    trashcan: false,
});

$.get({
    url: '/blocks',
    success: function(xml) {
        Blockly.Xml.domToWorkspace(Blockly.Xml.textToDom(xml), workspace);
    }
});

function saveBlock() {
    $.post({
        url: '/blocks',
        data: {"xml": Blockly.Xml.domToText(Blockly.Xml.workspaceToDom(workspace))},
    });    
}

function postModel(data) {
    //if( !$(this).val() ) {
    //    $(".alert").alert();   
    //}
    data["cs4ri_id"] = $("#cs4ri_id").val();
    console.log(data);
    $.post(
        '/train',
        JSON.stringify(data),
        function(res) {
            //console.log(res);
            //alert( "success" );
            $('#imageDiv').html('<img src="data:image/png;base64,'+res+'" />');
        }
        //'json'
    );
    //.done(function() {
    //    alert( "second success" );
    //})
    //.fail(function() {
    //    alert( "error" );
    //})
    //.always(function() {
    //    alert( "finished" );
    //});

}

function testModel() {
    var code = Blockly.JavaScript.workspaceToCode(workspace);
    console.log(code);
    try {
        eval(code);
    } catch (e) {
        //alert(e); // TODO: bootstrap alert
    }
}

function submitEntry() {
    
    $.post(
        '/forecast',
        JSON.stringify({"cs4ri_id": $("#cs4ri_id").val()}),
        function(res) {
            //console.log(res);
            //alert( "success" );
            $('#forecastImageDiv').html('<img src="data:image/png;base64,'+res+'" />');
            $('#model_cs4ri_id').html($("#cs4ri_id").val());
            $('#forecast_modal').modal('show')
        }
    );
}

