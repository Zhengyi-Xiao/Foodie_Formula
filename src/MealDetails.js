import React from 'react';
import './MealDetails.css';

const MealDetails = ({ meal, setMeal, file, setFile }) => {
  let totalCalories = 0;
  let totalWater = 0;
  let totalEnergy = 0;
  let totalVitamins = 0;
  let totalMinerals = 0;
  let totalProtein = 0;
  let totalCarbohydrates = 0;
  let totalFats = 0;

  const maxCalories = 2000;
  const maxWater = 1000;
  const maxEnergy = 10000;
  const maxVitamins = 1000;
  const maxMinerals = 1000;
  const maxProtein = 100;
  const maxCarbohydrates = 300;
  const maxFats = 100;

  meal["nutrients"].forEach(item => {
    totalCalories += item.amountCalories || 0;
    totalWater += item.amountWater || 0;
    totalEnergy += item.amountEnergy || 0;
    totalVitamins += item.hasNutrient.vitamins || 0;
    totalMinerals += item.hasNutrient.minerals || 0;
    totalProtein += item.hasNutrient.protein || 0;
    totalCarbohydrates += item.hasNutrient.carbohydrates || 0;
    totalFats += item.hasNutrient.fats || 0;
  });

  const handleHomeClick = () => {
    setMeal(null);
    setFile(null);
  };

  return (
    <div className="meal-container">
      <div className='meal-header'>
        <div className="meal-image-container">
          {file ? (
            <img src={file} alt="Meal Preview" className="meal-image" />
          ) : (
            <div className="meal-image-placeholder"></div>
          )}
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
          <span className="nutrition-label">Calories</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalCalories / maxCalories) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalCalories.toFixed(2)} Cal</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Water</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalWater / maxWater) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalWater.toFixed(2)} g</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Energy</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalEnergy / maxEnergy) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalEnergy.toFixed(2)} kJ</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Vitamins</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalVitamins / maxVitamins) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalVitamins.toFixed(2)} mcg</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Minerals</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalMinerals / maxMinerals) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalMinerals.toFixed(2)} mg</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Protein</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalProtein / maxProtein) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalProtein.toFixed(2)} g</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Carbs</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalCarbohydrates / maxCarbohydrates) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalCarbohydrates.toFixed(2)} g</span>
        </div>
        <div className="nutrition-row">
          <span className="nutrition-label">Fats</span>
          <div className="nutrition-bar">
            <div className="nutrition-bar-value" style={{ width: `${(totalFats / maxFats) * 100}%` }}></div>
          </div>
          <span className="nutrition-value">{totalFats.toFixed(2)} g</span>
        </div>
      </div>
      <h3 className="section-title">Nutritional Analysis</h3>
      <div className="suggestions">
        {meal["suggestions"]}
      </div>
      <button className="home-button" onClick={handleHomeClick}>Home</button>
    </div>
  );
};

export default MealDetails;
