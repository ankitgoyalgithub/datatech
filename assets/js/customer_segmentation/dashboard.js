/* globals Chart:false, feather:false */
(function () {
    'use strict'
    feather.replace();

    google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable([
          ['Clusters', '%'],
          ['C1',35],
          ['C2',52],
          ['C3',13]
        ]);
        var options = {
          title: 'Cluster Details',
          is3D: true,
          chartArea: {
            left:"20%"
          }
        };
        var chart = new google.visualization.PieChart(document.getElementById('piechart'));
        chart.draw(data, options);
      }
  }())