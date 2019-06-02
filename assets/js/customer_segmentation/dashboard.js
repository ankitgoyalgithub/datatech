/* globals Chart:false, feather:false */
(function () {
	'use strict'
    feather.replace();
	google.charts.load('current', {'packages':['corechart']});
	google.charts.setOnLoadCallback(drawChart);
	function drawChart() {
		$.ajax({
			method: "GET",
			url: "/cseg/cluster-distribution",
			contentType: "application/json"
		}).done(function(msg) {
			let dataArray = [];
			dataArray.push(['Clusters', '%'])

			for(let key in msg){
				dataArray.push([key, msg[key]])
			}
			
			var data = google.visualization.arrayToDataTable(dataArray);
			var options = {
				title: 'Cluster Details',
				is3D: true,
				chartArea: {
					left:"20%"
				}
			};
			var chart = new google.visualization.PieChart(document.getElementById('piechart'));
			chart.draw(data, options);
		});
	}
}())

$(document).ready(function(){

	/* Cluster Details Ajax Call */
	$.ajax({
		method: "GET",
		url: "/cseg/cluster-details",
		contentType: "application/json"
	}).done(function(msg) {
		$('#dashboard-table-head').html('');
		$('#dashboard-table-rows').html('');
		let tableHeadStr = "";
		for(let key in msg["columns"]){
			tableHeadStr = tableHeadStr + '<th scope="col">' + msg["columns"][key] + '</th>'
		}
		$('#dashboard-table-head').append(
			'<tr>' + tableHeadStr + '</tr>'
		)
		for(let key in msg["rows"]){
			$('#dashboard-table-rows').append(
				'<tr>' +
					'<th scope="row">'+msg['rows'][key]['LABEL']+'</th>' +
					'<td>'+msg['rows'][key]['COUNT']+'</td>' +
					'<td>'+msg['rows'][key]['AVERAGE_TIME_ON_PAGE']+'</td>' +
					'<td>'+msg['rows'][key]['EXITS']+'</td>' +
					'<td>'+msg['rows'][key]['PAGEVIEWS']+'</td>' +
					'<td>'+msg['rows'][key]['AVERAGE_TIME_ON_PAGE_LEVEL']+'</td>' +
					'<td>'+msg['rows'][key]['EXITS_LEVEL']+'</td>' +
					'<td>'+msg['rows'][key]['PAGEVIEWS_LEVEL']+'</td>' +
				'</tr>'
			)
		}
	});

	/* On Dashboard Reload */
	$('#cluster-insights-button').click(function() {
		location.reload();
	});

	/* Account Insights Button */
	$("#account-insights-button").click(function(event){
		$('#dashboard-table-head').html('');
		$('#dashboard-table-rows').html('');
		$('#piechart').hide();
		$.ajax({
			method: "GET",
			url: "/cseg/account-insights",
			contentType: "application/json"
		}).done(function(msg) {
			let tableHeadStr = "";
			for(let key in msg["columns"]){
				tableHeadStr = tableHeadStr + '<th scope="col">' + msg["columns"][key] + '</th>'
			}
			$('#dashboard-table-head').append(
				'<tr>' + tableHeadStr + '</tr>'
			)
			for(let key in msg["rows"]){
				$('#dashboard-table-rows').append(
					'<tr>' +
						'<th scope="row">'+msg['rows'][key]['LABEL']+'</th>' +
						'<td>'+msg['rows'][key]['ACCOUNTID']+'</td>' +
						'<td>'+msg['rows'][key]['AVERAGE_TIME_ON_PAGE']+'</td>' +
						'<td>'+msg['rows'][key]['PAGEVIEWS']+'</td>' +
						'<td>'+msg['rows'][key]['CLUSTER']+'</td>' +
						'<td>'+msg['rows'][key]['PAGEVIEWS_LEVEL']+'</td>' +
						'<td>'+msg['rows'][key]['EXITS_LEVEL']+'</td>' +
						'<td>'+msg['rows'][key]['MESSAGE']+'</td>' +
					'</tr>'
				)
			}
		});
	});
});