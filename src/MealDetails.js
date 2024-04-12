import React from 'react';
import './MealDetails.css';

const MealDetails = ({ meal, setMeal, file, setFile }) => {
  let totalCalories = 0;
  let totalWater = 0;
  let totalEnergy = 0;
  let nutrients = {};
  const maxCalories = 2000;
  const maxNutrientValue = 1000;
  const maxEnergy = 10000;

  meal["nutrients"].forEach(item => {
    totalCalories += item?.amountCalories || 0;
    totalWater += item?.amountWater || 0;
    totalEnergy += item?.amountEnergy || 0;

    // for (const [key, value] of Object.entries(item?.hasNutrient)) {
    //   nutrients[key] = (nutrients[key] || 0) + value;
    // }
  });

  const handleHomeClick = () => {
    setMeal(null);
    setFile(null);
  }

  return (
    <div className="meal-container">
      <div className='meal-header'>
        <div className="meal-image-container">
          {file
            ? <img src={file} alt="Meal Preview" className="meal-image" />
            : <div className="meal-image-placeholder"></div>
          }
        </div>
        <h2 className="meal-title">Meal 1</h2>
      </div>
      <div className="ingredients">
        {meal["nutrients"].map((ingredient, index) => (
          <span key={index} className="ingredient">{ingredient.item}</span>
        ))}
      </div>
      <h3 className="section-title">Nutritional Information</h3>
      <div className="nutrition-info">
        <div className="nutrition-row">
          <span className="nutrition-label">Carbs</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalCalories / maxCalories) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalCalories.toFixed(2)} Cal</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Fat</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalWater / maxNutrientValue) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalWater.toFixed(2)} g</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Protein</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalEnergy / maxEnergy) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalEnergy.toFixed(2)} g</span>
        </div>
      </div>
      <h3 className="section-title">Nutritional Analysis</h3>
      {meal["suggestions"]}
      <button className="home-button" onClick={handleHomeClick}>Home</button>
    </div>
  );
};

export default MealDetails;
