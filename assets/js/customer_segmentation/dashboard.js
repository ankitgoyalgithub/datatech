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
	$.ajax({
		method: "GET",
		url: "/cseg/cluster-details",
		contentType: "application/json"
	}).done(function(msg) {
		for(let key in msg){
			$('#cluster-details').append(
				'<tr>' +
					'<th scope="row">'+msg[key]['LABEL']+'</th>' +
					'<td>'+msg[key]['COUNT']+'</td>' +
					'<td>'+msg[key]['AVERAGE_TIME_ON_PAGE']+'</td>' +
					'<td>'+msg[key]['EXITS']+'</td>' +
					'<td>'+msg[key]['PAGEVIEWS']+'</td>' +
					'<td>'+msg[key]['AVERAGE_TIME_ON_PAGE_LEVEL']+'</td>' +
					'<td>'+msg[key]['EXITS_LEVEL']+'</td>' +
					'<td>'+msg[key]['PAGEVIEWS_LEVEL']+'</td>' +
				'</tr>'
			)
		}
	});
});