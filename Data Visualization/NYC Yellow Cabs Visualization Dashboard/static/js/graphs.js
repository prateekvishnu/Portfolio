queue()
    .defer(d3.json, "/data")
    .await(makeGraphs);

function makeGraphs(error, recordsJson) {
    
    //Clean data
    var records = recordsJson;
    var dateFormat = d3.time.format("%Y-%m-%d %H:%M:%S");
    
    records.forEach(function(d) {
        d["pickup_datetime"] = dateFormat.parse(d["pickup_datetime"]);
        d["pickup_datetime"].setMinutes(0);
        d["pickup_datetime"].setSeconds(0);
        d["pickup_longitude"] = +d["pickup_longitude"];
        d["pickup_latitude"] = +d["pickup_latitude"];
    });

    //Create a Crossfilter instance
    var ndx = crossfilter(records);

    //Define Dimensions
    var dateDim = ndx.dimension(function(d) { return d["pickup_datetime"]; });
    var totalfareDim = ndx.dimension(function(d) { return d["total_amount_segment"]; });
    var tipSegmentDim = ndx.dimension(function(d) { return d["tip_amount_segment"]; });
    var passengerCountDim = ndx.dimension(function(d) { return d["passenger_count_segment"]; });
    var dim1 = ndx.dimension(function (d) {
        return [+d["tip_amount_segment"], +d["total_amount_segment"]];
    });

    var locationdDim = ndx.dimension(function(d) { return d["pickup_borough"]; });
    var tiondDim = ndx.dimension(function(d) { return d["dropoff_borough"]; });
    var allDim = ndx.dimension(function(d) {return d;});


    //Group Data
    var numRecordsByDate = dateDim.group();
    var totalfareGroup = totalfareDim.group();
    var tipSegmentGroup = tipSegmentDim.group();
    var passengerCountGroup = passengerCountDim.group();
    var locationGroup = locationdDim.group();
    //var tionGroup = tiondDim.group();
    var group1 = dim1.group();
    var all = ndx.groupAll();


    //Define values (to be used in charts)
    var minDate = dateDim.bottom(1)[0]["pickup_datetime"];
    var maxDate = dateDim.top(1)[0]["pickup_datetime"];


    //Charts
    var numberRecordsND = dc.numberDisplay("#number-records-nd");
    var timeChart = dc.barChart("#time-chart");
    var totalfareChart = dc.rowChart("#totalfare-row-chart");
    var tipSegmentChart = dc.rowChart("#tip-segment-row-chart");
    var passengerCountChart = dc.rowChart("#passenger-count-row-chart");
    var locationChart = dc.pieChart("#location-row-chart");
    //var tionChart = dc.rowChart("#tion-row-chart");     
    numberRecordsND
        .formatNumber(d3.format("d"))
        .valueAccessor(function(d){return d; })
        .group(all);


    timeChart
        .width(1700)
        .height(140)
        .margins({top: 10, right: 250, bottom: 20, left: 250})
        .dimension(dateDim)
        .group(numRecordsByDate)
        .transitionDuration(100)
        .x(d3.time.scale().domain([minDate, maxDate]))
        .elasticY(true)
        .yAxis().ticks(3);
        //.xAxisLabel("Fare");

    totalfareChart
        .width(400)
        .height(220)
        .dimension(totalfareDim)
        .group(totalfareGroup)
        .ordering(function(d) { return -d.value })
        .colors(['#6baed6'])
        .elasticX(true)
        .xAxis().ticks(5);

    tipSegmentChart
        .width(400)
        .height(220)
        .dimension(tipSegmentDim)
        .group(tipSegmentGroup)
        .colors(['#6baed6'])
        .elasticX(true)
        .labelOffsetY(10)
        .xAxis().ticks(5);

    passengerCountChart
        .width(400)
        .height(220)
        .dimension(passengerCountDim)
        .group(passengerCountGroup)
        .ordering(function(d) { return -d.value })
        .colors(['#6baed6'])
        .elasticX(true)
        .xAxis().ticks(5);



   /* TipVsFareChart
    .width(300)
    .height(300)
    .x(d3.scale.linear().domain([0, 20]))
    .yAxisLabel("y")
    .xAxisLabel("x")
    .clipPadding(10)
    .dimension(dim1)
    .excludedOpacity(0.5)
    .group(group1);    
*/
     locationChart
         .width(400)
         .height(220)
         .dimension(locationdDim)
         .group(locationGroup)
         .radius(100)
         .colors(d3.scale.ordinal().range(['#d0d1e6','#a6bddb','#74a9cf','#2b8cbe','#045a8d']));

         //.ordering(function(d) { return -d.value })
         //.colors(['#6baed6'])
         //.elasticX(true)
         //.elasticY(true);
         //.labelOffsetY(10)
         //.xAxis().ticks(4);

   /* tionChart
         .width(300)
         .height(410)
         .dimension(tiondDim)
         .group(tionGroup)
         .ordering(function(d) { return -d.value })
         .colors(['#6baed6'])
         .elasticX(true)
         .labelOffsetY(10)
         .xAxis().ticks(4);
*/

    var map = L.map('map');
 

    var drawMap = function(){

        map.setView(new L.LatLng(40.737, -73.923), 10);
        mapLink = '<a href="http://openstreetmap.org">OpenStreetMap</a>';
        L.tileLayer(
            'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; ' + mapLink + ' Contributors',
                maxZoom: 15,
            }).addTo(map);

        //HeatMap
        var geoData = [];
        _.each(allDim.top(Infinity), function (d) {
            geoData.push([d["pickup_latitude"], d["pickup_longitude"], 1]);
          });
        var heat = L.heatLayer(geoData,{
            radius: 10,
            blur: 20, 
            maxZoom: 3 ,
        }).addTo(map);

    };

    //Draw Map
    drawMap();

    //Update the heatmap if any dc chart get filtered
    dcCharts = [timeChart,tipSegmentChart,passengerCountChart,totalfareChart,locationChart];
    _.each(dcCharts, function (dcChart) {
        dcChart.on("filtered", function (chart, filter) {
            map.eachLayer(function (layer) {
                map.removeLayer(layer)
            }); 
            drawMap();
        });
    });

    dc.renderAll();

};