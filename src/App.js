import React, { useState } from 'react';

import FileUpload from './FileUpload';
import MealDetails from './MealDetails';
function App() {
  const [meal, setMeal] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');

  return (
    <div className="App">
      {meal ? (
        <MealDetails meal={meal} setMeal={setMeal} file={imagePreviewUrl} setFile={setImagePreviewUrl} />
      ) : (
        <FileUpload setMeal={setMeal} setFile={setImagePreviewUrl} />
      )}
    </div>
  );
}

export default App;
