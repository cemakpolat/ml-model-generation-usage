// app.js

$(document).ready(function() {
    // Define the Flask API endpoint
    const apiEndpoint = 'http://localhost:8080/predict';

    // Handle button click event
    $('#predictButton').on('click', function() {
        // Get input values
        const humidity = $('#humidity').val();
        const windSpeed = $('#windSpeed').val();
        const pressure = $('#pressure').val();
        const windBearing = $('#windBearing').val();
        const visibility = $('#visibility').val();

        // Make a POST request to the Flask API
        $.ajax({
            type: 'POST',
            url: apiEndpoint,
            contentType: 'application/json;charset=UTF-8',
            data: JSON.stringify({ data: [humidity, windSpeed, pressure, windBearing, visibility] }),
            success: function(response) {
                // Update the result on the page
                 $('#predictionResult').html('<strong>Prediction Result:</strong> ' + response.prediction);
            },
            error: function(error) {
                console.error('Error making API request:', error);
            }
        });
    });
});