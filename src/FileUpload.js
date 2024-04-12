import React from 'react';
import './FileUpload.css';
import Camera from "./camear.svg";
import axios from 'axios'; // Don't forget to import axios


function FileUpload({ setMeal, setFile }) {
  // Function to handle file selection
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.match('image.*')) {
      const reader = new FileReader();

      reader.onloadend = () => {
        setFile(reader.result);
      };

      reader.readAsDataURL(file);
    }
    if (file) {
      try {
        console.log('Uploading file:', file);
        const formData = new FormData();
        formData.append('image', file); // Assuming 'file' is the key expected by the backend

        const response = await axios({
          method: 'post',
          url: 'http://0.0.0.0:8000/uploadImage',
          data: formData,
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        const result = response.data;
        console.log('Upload result:', result);
        setMeal(result); // Update state or do something with the result
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    } else {
      alert('Please select a file first!');
    }
  };

  const handleClick = () => {
    // When the viewfinder is clicked, click the hidden file input
    document.getElementById('file-upload').click();
  };

  return (
    <div>
      <div className="camera-container">
        <h1>Take a picture of your meal!</h1>
        <p>Please make sure your fork is included in the picture.</p>
        <button className="camera-viewfinder" onClick={handleClick}>
        </button>
        <button className="camera-button" onClick={handleClick}>
          <img className="camera-icon" src={Camera} alt="React Logo" />
        </button>
        <input
          type="file"
          style={{ display: 'none' }}
          id="file-upload"
          onChange={handleFileChange}
        />
      </div>
    </div>
  );
}

export default FileUpload;
