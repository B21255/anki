var filename='';
var resultFileName = '';
var resultFilePath ='';

function hideAllScreen() {
	$('#home_screen').hide();
	$('#upload_screen').hide();
    $('#about_us_screen').hide();
}

function showScreen(divName) {
	$('#'+ divName).show();
}

$(document).ready(function() {  
    $('a').click(function(){
        var tabId = $(this).attr('id');
        var screenId= tabId.substr(3, tabId.length);
        if(tabId.substr(0,3)=='src'){
            hideAllScreen();
            showScreen(screenId);
            
           } 
    });
});

function disableScreen() {
    $('#srcmodel_run_screen').attr('disabled','disabled');
}

function uploadFile() {
$('#loader-wrapper').show();
    console.log("in java script");
    var formdata, file, seq_data;
    formdata = new FormData();
    if($('#fileSelect').val()){
    file = $('#fileSelect').get(0).files[0];
        formdata.append('file',file);
        console.log("the file: "+ formdata.get('file'));
    }
    else{
     $('#loader-wrapper').hide();
     $('#uploadResponse').show();
     $('#uploadResponse').html("No data uploaded");}
    $.ajax({
        url:"/upload",
        type:"POST",
        data: formdata,
        processData: false,
        contentType: false,
        success: function(response){
            var obj = JSON.parse(response);
            if(obj.status=='OK'){

            filename = obj.file_name;
            var formdata_2 = new FormData();
            formdata_2.append('file_name', filename);
                $.ajax({
                url:"/model",
                type:"POST",
                data: formdata_2,
                processData: false,
                contentType: false,
                success: function(response){
                        var obj_2 = JSON.parse(response);
                        $('#upload_heading').text('');
                        $('#loader-wrapper').hide();
                        $('#uploadText').hide();
                        $('#upload_data_form').hide();
                        $('#myTable').show();


                    // wrting the data in the html table
                        for (var i=0; i<obj_2.length; i++) {
                            var row = $('<tr><td>' + (i+1)+ '</td><td style="text-align:centre;">' +
                            obj_2[i].Sequence_ID+ '</td><td style="text-align:justify;">' +
                            obj_2[i].sequences + '</td><td style="text-align:centre;">' +
//
//                            obj_2[i].CNN_LSTM + '</td><td style="text-align:centre;">'+
//                            obj_2[i].Heat_Stress + '</td><td style="text-align:centre;">' +
                            obj_2[i].preds + '</td></tr>');
                            $('#myTable').append(row);
        }
                            $('#loader-wrapper').hide();
                },
                error : function(){
                        $('#loader-wrapper').hide();
                        $('#uploadResponse2').show();
                        $('#uploadResponse2').html("Oops!! some error occured");
                        console.log("AJAX not working!");
                    }
            });
            }
            else if(obj.status=='NOT CSV'){
                $('#loader-wrapper').hide();
                $('#uploadResponse').show();
                $('#uploadResponse').html("please upload a csv file!!!");
            }
            else{
                console.log("status "+ obj.status);
                $('#loader-wrapper').hide();
                $('#uploadResponse').show();
                $('#uploadResponse').html("Oops!!! Dataset file not upoaded.");
            }
        },
 		error : function(){
                $('#loader-wrapper').hide();
                $('#uploadResponse').show();
                $('#uploadResponse').html("Oops!!! Dataset file not upoaded, some error has occured.");
				console.log("AJAX not working!");
	    }
    });
}

function runModel(){
    console.log("in java script");
    var formdata, modeltype, acttype;
    formdata = new FormData();

    if($('#modelType :selected').text()=='--select--' && $('#activationType :selected').text()=='--select--' ){
            $('#mod_error').show();
            $('#mod_error').html("* select a model type");
            $('#act_error').show();
            $('#act_error').html("* select an activation function");
            }

    else if($('#modelType :selected').text()=='--select--'){
        $('#act_error').hide();
        $('#mod_error').show();
        $('#mod_error').html("* select a model type");

    }
    else if($('#activationType :selected').text()=='--select--'){
        $('#mod_error').hide();
        $('#act_error').show();
        $('#act_error').html("* select an activation function");

    }
    else {
        $('#loader-wrapper').show();
        $('#mod_error').hide();
        $('#act_error').hide();
        modeltype = $('#modelType :selected').text();
        acttype = $('#activationType :selected').text();

        formdata.append('file_name', filename);
        formdata.append('model_type',modeltype);
        formdata.append('act_type', acttype);

        console.log("the file: "+ formdata.get('file_name'));
        console.log("the model: "+ formdata.get('model_type'));
        console.log("the activation: "+ formdata.get('act_type'));

   }
}

hideAllScreen();
disableScreen();
showScreen('home_screen');
