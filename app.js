var express = require('express');
var app = express();

app.engine('html', require('ejs').renderFile);


// Routes
app.get('/', function(req, res) {
	res.render('home.html');
});

app.get('/report', function(req, res) {
	res.render('report.html');
});

app.get('*', function(req, res) {
	res.redirect('/');
});



// Listen for requests on port 3000
app.listen(3000, 'localhost', function() {
	console.log("Server listening on port 3000...");
});