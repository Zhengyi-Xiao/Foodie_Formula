import React, { useState } from 'react';
import './FileUpload.css';
import Camera from "./camear.svg";


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
      const formData = new FormData();
      formData.append('file', file);

      try {
        // const response = await fetch('http://localhost:8000/uploadimage', {
        //   method: 'POST',
        //   body: formData,
        // });

        // Mock response
        // const result = await response.json();
        const result = {
          "nutrients": [{
            'item': 'strawberry',
            'amountCalories': 27.0,
            'amountEnergy': 113.0,
            'amountWater': 90.5,
            'ediblePart': 94.0,
            'hasNutrient': {},
            'foodId': 411.0
          },
          {
            'item': 'chicken',
            'amountCalories': 175.0,
            'amountEnergy': 732.0,
            'amountWater': 68.7,
            'ediblePart': 68.0,
            'hasNutrient': {},
            'foodId': 1098.0
          },
          {
            'item': 'potato',
            'amountCalories': 110.0,
            'amountEnergy': 460.3,
            'ediblePart': 100.0,
            'hasNutrient': {},
            'foodId': 8.0,
            'isUnspecifiedFood': 1.0
          },
          {
            'item': 'tomato',
            'amountCalories': 17.0,
            'amountEnergy': 71.0,
            'amountWater': 94.2,
            'ediblePart': 100.0,
            'hasNutrient': {},
            'foodId': 391.0
          },
          {
            'item': 'lettuce',
            'amountCalories': 19.0,
            'amountEnergy': 79.0,
            'amountWater': 94.3,
            'ediblePart': 80.0,
            'hasNutrient': {},
            'foodId': 331.0
          },
          {
            'item': 'cilantro mint',
            'amountCalories': 393.0,
            'amountEnergy': 1644.0,
            'amountWater': 0.2,
            'ediblePart': 100.0,
            'hasNutrient': {},
            'foodId': 19012.0
          }], "suggestions": "This meal packs protein, carbs, vitamins, and minerals but watch out for heart-unfriendly fats from fries, steak, and pork! To keep it balanced, try swapping some steak and pork for lean proteins like fish, chicken. Add vibrant fruits and veggies for essential nutrients and swop fries for healthier baked potatoes. Happy eating, heartily and healthily!"
        }
        console.log(result); // Log the mock response
        setMeal(result);
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
