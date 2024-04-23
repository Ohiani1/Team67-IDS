function importPopup() {
    document.getElementById("myImportPopup").style.display = "block";
}


function closeImportPopup() {
    document.getElementById("myImportPopup").style.display = "none";
}


function infoPopup() {
    document.getElementById("myMoreInformation").style.display = "block";
}


function closeInfoPopup() {
    document.getElementById("myMoreInformation").style.display = "none";
}

document.getElementById('model').addEventListener('change', function() {
  var selectedValue = this.value;
  var parametersContainer = document.querySelector('.parameters_container');
  
  if (selectedValue === 'MHT') {
    parametersContainer.style.display = 'flex';
  } else {
    parametersContainer.style.display = 'none';
  }
});


function scrollToCompareResults() {
  const compareResultsDiv = document.getElementById('compare-results');
  compareResultsDiv.scrollIntoView({ behavior: 'smooth' });
}
let called = 0;
let updatedb = false;

let metricsData = null;
let currentModel = null;
let currentDataSet = null

var homeUrl = "https://27a1-153-33-34-246.ngrok-free.app"



async function callBackendAPI() {
    const loadingCircle = document.getElementById('loading-circle');
    loadingCircle.classList.remove('hidden');
    try {
      // Get the selected option value from the dropdown
      const selectedOption = document.getElementById("model").value;
      currentModel = selectedOption
      const selectedDataset = document.getElementById("dataset").value;
      currentDataSet = selectedDataset
  
      // Update the button text to the selected option
      document.querySelector('.runButton').textContent = selectedOption;
  
      // Construct the URL with selected option as path parameter
      const url = `${homeUrl}/run/${encodeURIComponent(selectedOption)}/${encodeURIComponent(selectedDataset)}`;
  
      // Send a GET request to the backend API
      const response = await fetch(url, { mode: 'no-cors' });
  
      if (response.ok) {
        const data = await response.json();
        loadingCircle.classList.add('hidden');
        console.log('Response from backend (GET):', data);
        const metrics = data[selectedOption]
        metricsData = metrics;

        document.getElementById("precision_api").textContent = metrics.Precision
        document.getElementById("recall_api").textContent = metrics.Recall
        document.getElementById("accuracy_api").textContent = metrics.Accuracy
        document.getElementById("f1_api").textContent = metrics.F1_score

        document.getElementById("compare_precision").textContent = metrics.Precision
        document.getElementById("compare_recall").textContent = metrics.Recall
        document.getElementById("compare_accuracy").textContent = metrics.Accuracy
        document.getElementById("compare_f1").textContent = metrics.F1_score
        //run = data; // Do something with the response from the backend if needed
      } else {
        const text = await response.text();
        loadingCircle.classList.add('hidden');
        console.error(`Network response was not ok (${response.status}): ${text}`);
        document.querySelector('.runButton').textContent = "fail";
        throw new Error(`Network response was not ok (${response.status}): ${text}`);
      }
    } catch (error) {
        loadingCircle.classList.add('hidden');
      console.error('There was a problem with your fetch operation:', error);
      document.querySelector('.runButton').textContent = "fail";
    }
  }

document.getElementById("model").addEventListener("change", function() {
    var selectedOption = this.value;
    // Now you can use the selectedOption variable in your JavaScript code
    console.log("Selected option:", selectedOption);
    // You can call any function or perform any action with the selected option here
    // For now, let's update the "Run Test" button text to match the selected option
    document.querySelector('.runButton').textContent = selectedOption;
});

async function uploadDataToServer() {
  const loadingCircle = document.getElementById('loading-circle');
  loadingCircle.classList.remove('hidden');

  if (metricsData && currentDataSet && currentModel)
  {
    // Define the API endpoint URL
    const uploadApiUrl = `${homeUrl}/save`;
    const paylod = {
      model: currentModel,
      dataset: currentDataSet,
      metrics: metricsData
    }

    try {
      // Send a POST request with the data payload
      const response = await fetch(uploadApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(paylod)
      });

      if (response.ok) {
        loadingCircle.classList.add('hidden');
        updatedb = true
        // Data was successfully uploaded
        console.log('Data uploaded successfully');
      } else {
        loadingCircle.classList.add('hidden');
        // Error occurred while uploading data
        console.error('Error uploading data:', response.status);
      }
    } catch (error) {
      loadingCircle.classList.add('hidden');
      // Handle network errors
      console.error('Network error:', error);
    }
  }
  else {
    console.log('No data available to upload.');
  }
  
}

async function fetchRuns() {
  if (called == 0 || updatedb == true)
  {
    try {
      // Fetch runs from the database (replace the URL with your actual API endpoint)
      const response = await fetch(`${homeUrl}/prevruns`);
      const data = await response.json();
      
      // Get the select element
      const selectElement = document.getElementById('test');
  
      selectElement.innerHTML = '<option value="example">select test</option>';
  
      
      // Add each run as an option to the select element
      data.forEach(run => {
        const option = document.createElement('option');
        option.value = run.model
        option.textContent = `${run.model} | ${run.dataset} | ${new Date(run.timestamp.$date).toLocaleString()}`
        selectElement.appendChild(option);
      });
    } catch (error) {
      console.error('Error fetching runs:', error);
    }
  }
  called++;
  updatedb = false;
  
}

// Add event listener to the select element
if (called == 0 || updatedb == true)
{
  document.getElementById('test').addEventListener('click', fetchRuns);
}






// function runTest() {
//     var scriptPath = "C:/Users/baseb/Desktop/IDS Workspace/Main.py";
//     subprocess.run(["python", scriptPath]);
// } 

//function sends input information to the backend


// document.addEventListener("DOMContentLoaded", function() {
//     // Add event listener to the dropdown select element
//     const modelSelect = document.getElementById("model");
//     modelSelect.addEventListener("change", async function() {
//         // Get the selected model
//         const selectedModel = modelSelect.value;

//         // Display "loading" message (assuming you have an element with ID "status")
//         const statusElement = document.getElementById("status");
//         statusElement.textContent = "Loading...";

//         try {
//             // Call the backend API function with the selected model
//             const response = await callBackendAPI(selectedModel);

//             // Update status to "Success" upon successful response
//             statusElement.textContent = "Success!";

//             // Process the API response (assuming you have a function to handle it)
//             handleAPIResponse(response);
//         } catch (error) {
//             // Handle errors if the API call fails
//             console.error("Error:", error);
//             statusElement.textContent = "Error"; // Update status to indicate error
//         }
//     });
// });

// async function callBackendAPI(selectedModel) {
//     // Make sure 'selectedModel' is one of the valid options
//     if (selectedModel !== "LCCDE" && selectedModel !== "MHT" && selectedModel !== "Decision Tree") {
//         console.error("Invalid model selection.");
//         return;
//     }

//     // Define the URL for the backend API
//     const url = `/run/${selectedModel.toLowerCase()}`;

//     // Make a GET request to the backend API
//     try {
//         const response = await fetch(url);
//         // Check if the response is successful (status code 200)
//         if (response.ok) {
//             return await response.json();
//         } else {
//             throw new Error('Failed to fetch data from the server.');
//                button.textContent = "Success";

//         }
//     } catch (error) {
//         throw error; // Re-throw the error for handling in the main function
//     }
// }

// function handleAPIResponse(data) {
//     // Process the data received from the server (replace with your logic)
//     console.log(data);
// }




































// function callBackendAPI(selectedModel) {
//     // Get the button element
//     const button = document.getElementById("runButton");

//     // Change button text to "Loading..."
//     button.textContent = "Loading...";

//     // Make sure 'selectedModel' is one of the valid options
//     if (selectedModel !== "LCCDE" && selectedModel !== "MHT" && selectedModel !== "Decision Tree") {
//         console.error("Invalid model selection.");
//         // Reset button text to "Run Test" if an invalid model is selected
//         button.textContent = "Run Test";
//         return;
//     }

//     // Define the URL for the backend API
//     const url = `/run/${selectedModel.toLowerCase()}`;

//     // Make a GET request to the backend API
//     fetch(url)
//         .then(response => {
//             // Check if the response is successful (status code 200)
//             if (response.ok) {
//                 // Parse the response body as JSON
//                 return response.json();
//             } else {
//                 // Handle errors if the response is not successful
//                 throw new Error('Failed to fetch data from the server.');
//                 button.textContent = "no";

//             }
//         })
//         .then(data => {
//             // Handle the data received from the server
//             console.log(data); // You can process the result here
            
//             // Display success message
//             displaySuccessMessage();

//             // Change button text to "Success"
//             button.textContent = "Success";
//         })
//         .catch(error => {
//             // Handle any errors that occurred during the fetch
//             console.error('Error:', error);
//             // Reset button text to "Run Test" in case of an error
//             button.textContent = "Run Test";
//         });
// }