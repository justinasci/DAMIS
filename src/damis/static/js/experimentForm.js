;
(function() {
	window.experimentForm = {
		// parameters for window.experimentForm initialization
		params: {},

		// translate parameter binding from client to server
		// representation
		bindingToServer: function() {
			$.each($(".task-window input[id$=connection_type]"), function() {
				if ($(this).val() === "INPUT_CONNECTION") { // inspect each input parameter
					var srcRefField = $(this).closest("div").find("input[id$=source_ref]");
					var oParamAddr = $(srcRefField).val();
					if (oParamAddr) {
						var parts = oParamAddr.split(",");
						var oParam = window.experimentForm.getParameter(parts[0], parts[1]);
						var oParamField = window.experimentForm.getParameterValue(oParam);
						srcRefField.val(oParamField.attr("name"));
					}
				}
			});
		},

		// translate parameter binding from server to client
		// representation
		// parameterFormset - target box parameters
		bindingToClient: function(parameterFormset) {
			$.each(parameterFormset.find("input[id$=connection_type]"), function() {
				if ($(this).val() === "INPUT_CONNECTION") { // inspect each input parameter
					var srcRefField = $(this).closest("div").find("input[id$=source_ref]");
					var oParamName = $(srcRefField).val();
					if (oParamName) {
						var oParam = $("input[name=" + oParamName + "]");
						var sourceForm = oParam.closest(".task-window");
						var sourceBoxId = window.taskBoxes.getBoxId(sourceForm);

						var oParent = oParam.closest("div");
						var paramNo = oParent.index();
						srcRefField.val(paramNo + "," + sourceBoxId);
					}
				}
			});
		},

		// refresh parameter formset prefixes before submition
		// call the callback after prefixes refresh
		updatePrefixes: function(parameterPrefixesUrl, callback, params) {
			// pass current task forms prefixes to get parameter
			// formsets prefixes
			var taskFormPrefixes = []
			var taskIds = []
			$.each($(".task-window .task-form"), function(taskBoxIdx, taskForm) {
				var name = $(taskForm).find("input,select,textarea,label").attr("name");
				var taskFormPrefix = /tasks-\d+/g;
				taskFormPrefixes.push(taskFormPrefix.exec(name)[0]);

				var taskId = $(taskForm).find("input[id$=id]").val();
				taskIds.push(taskId ? taskId: "-");
			});
			$.ajax({
				url: parameterPrefixesUrl,
				data: {
					prefixes: taskFormPrefixes,
					taskIds: taskIds
				},
				context: $(this)
			}).done(function(parameterFormsetPrefixes) {
				// when a box is deleted, other boxes have their ids
				// updated,  however, parameter formsets prefixes are not updated
				// we need to do it manually 
				var paramPrefixes = parameterFormsetPrefixes.split(",");
				$.each($(".task-window .parameter-values"), function(taskBoxIdx, paramsFormset) {
					$.each($(paramsFormset).find("input,select,textarea,label"), function(inputIdx, input) {
						var origPrefix = paramPrefixes[taskBoxIdx];
						var name = $(input).attr("name");
						var id = $(input).attr("id");
						if (name) {
							$(input).attr("name", name.replace(/PV_\d+/, origPrefix));
						}
						if (id) {
							$(input).attr("id", id.replace(/PV_\d+/, origPrefix));
						}
					});
				});
				callback(params);
			});
		},

		// Create form modal windows, assign them to boxes 
		reinitExperimentForm: function() {

			// recreate modal windows
			// iterate through existing task boxes
			// in the order of creation (asume, it is reflected
			// in DOM order)
			var updatedForms = $("#experiment-form .inline");
			$.each($(".task-box"), function(taskBoxId, taskBox) {
				taskForm = $(updatedForms[taskBoxId + 1]);
				parameterFormset = $(taskForm.next(".parameter-values"));
				// mark the task box as conataining errors
				if (parameterFormset.find(".errorlist").length > 0) {
					$(taskBox).addClass("error");
				} else {
					$(taskBox).removeClass("error");
				}
				window.taskBoxes.createTaskFormDialog(taskForm, parameterFormset, window.taskBoxes.getFormWindowId($(taskBox)));
				window.taskBoxes.addTaskBoxEventHandlers($(taskBox));
				window.taskBoxes.setBoxName($(taskBox).attr("id"));
			});
			$.each($(".task-box"), function(taskBoxId, taskBox) {
				//restore parameter bindings from server to client representation
				taskForm = $(updatedForms[taskBoxId + 1]);
				parameterFormset = $(taskForm.next(".parameter-values"));
				window.experimentForm.bindingToClient(parameterFormset);
			});
		},

		// Submits the experiment form and reinitializes it
		submit: function(params) {
			// translate parameter bindings from client to server
			// representation
			window.experimentForm.bindingToServer();

			var form = $("#experiment-form");
			if (params["skipValidation"]) {
				form.find("input[name=skip_validation]").val("True");
			}
			var data = form.serialize();
			$.post(form.attr("action"), data, function(resp) {
				if (!/<[a-z][\s\S]*>/i.test(resp)) {
					// non-html string is returned, which is a redirec url 
					window.location = resp;
					return;
				}
				//replace the existing form with the validated one
				$("#experiment-form").remove();
				$("#workflow-editor-container").before(resp);

				//run standard initialization
				window.experimentForm.init();
				window.experimentForm.reinitExperimentForm();
			});
		},

		// init formset plugin and form submit handlers
		init: function() {
			var params = window.experimentForm.params;

			parametersUrl = params['parametersUrl'];
			parameterPrefixesUrl = params['parameterPrefixesUrl'];
			taskFormPrefix = params['taskFormPrefix'];

			//initialize the jQuery formset plugin
			$('.inline').formset({
				prefix: taskFormPrefix,
				extraClasses: ['task-form'],
			});

			//assign form submit handler
			$('#save-btn').click(function(ev) {
				window.experimentForm.updatePrefixes(parameterPrefixesUrl, window.persistWorkflow.persist, {});
			});

			//assign new experiment handler
			$('#new-experiment-btn').click(function(ev) {
				window.location = params['experimentNewUrl'];
			});

			// open execute dialog
			$('#execute-btn').click(function(ev) {
				var dialog = $("#exec-dialog");
				if (!dialog.hasClass("ui-dialog-content")) {
					dialog.dialog({
						modal: true,
						appendTo: "#experiment-form",
						buttons: [{
							text: gettext('Cancel'),
							click: function(ev) {
								$(this).dialog("close");
							}
						},
						{
							text: gettext('Continue'),
							click: function(ev) {
								$(this).dialog("close");
								window.experimentForm.updatePrefixes(parameterPrefixesUrl, window.experimentForm.submit, {});
							}
						}],
						open: function() {
							$(this).closest(".ui-dialog").find("button").addClass('btn');
						}
					});
				} else {
					dialog.dialog("open");
				}
			});

		},

		// returns parameter form, given 
		// parameter number in the formset
		// and task box id
		getParameter: function(parameterNum, taskBoxId) {
			var taskFormWindow = $("#" + window.taskBoxes.getFormWindowId(taskBoxId));
			var paramForm = $(taskFormWindow.find(".parameter-values").find("div")[parameterNum]);
			return paramForm;
		},

		//returns parameter value field in parameter form
		getParameterValue: function(paramForm) {
			return paramForm.find("input[id$=value]");
		},

		//returns parameter source_ref field in parameter form
		getParameterSourceRef: function(paramForm) {
			return paramForm.find("input[id$=source_ref]");
		}

	}
})();

