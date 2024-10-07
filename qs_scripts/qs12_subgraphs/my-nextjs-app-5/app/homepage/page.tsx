import React from 'react';

const Homepage: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-4xl font-bold">Welcome to the Homepage</h1>
      <p className="mt-3 text-lg">
        This is the main entry point of the application.
      </p>
    </div>
  );
};

export default Homepage;