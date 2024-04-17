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

var run;

var homeUrl = "https://3a32-153-33-34-246.ngrok-free.app"



async function callBackendAPI() {
    try {
      // Get the selected option value from the dropdown
      const selectedOption = document.getElementById("model").value;
  
      // Update the button text to the selected option
      document.querySelector('.runButton').textContent = selectedOption;
  
      // Construct the URL with selected option as path parameter
      const url = `${homeUrl}/run/${encodeURIComponent(selectedOption)}`;
  
      // Send a GET request to the backend API
      const response = await fetch(url, { mode: 'no-cors' });
  
      if (response.ok) {
        const data = await response.json();
        console.log('Response from backend (GET):', data);
        //document.getElementById("response").innerHTML = "GET Response: " + JSON.stringify(data);
        run = data; // Do something with the response from the backend if needed
      } else {
        const text = await response.text();
        console.error(`Network response was not ok (${response.status}): ${text}`);
        document.querySelector('.runButton').textContent = "fail";
        throw new Error(`Network response was not ok (${response.status}): ${text}`);
      }
    } catch (error) {
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