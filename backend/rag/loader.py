"""
Data loader for car recommendations RAG system.
Loads Excel and CSV files, normalizes data, and converts types.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "processed"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_name(col: str) -> str:
    """Normalize column names to lowercase with underscores."""
    return col.lower().strip().replace(" ", "_").replace("-", "_")


def convert_numeric(value: Any) -> Any:
    """Convert value to numeric if possible, otherwise return original."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        # Remove currency symbols and commas
        if isinstance(value, str):
            cleaned = value.replace("$", "").replace(",", "").strip()
            if cleaned:
                return float(cleaned)
        return None
    except (ValueError, AttributeError):
        return None


def load_cars_excel(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load cars data from Excel file.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        List of normalized car records
    """
    logger.info(f"Loading cars data from {file_path}")
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Creating sample data.")
        return create_sample_cars_data()
    
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Normalize column names
        df.columns = [normalize_column_name(col) for col in df.columns]
        
        # Convert numeric columns
        numeric_columns = [
            col for col in df.columns 
            if any(keyword in col for keyword in ['price', 'cost', 'mpg', 'mileage', 'year', 'engine', 'horsepower', 'torque'])
        ]
        
        for col in numeric_columns:
            df[col] = df[col].apply(convert_numeric)
        
        # Convert to list of dicts
        records = df.to_dict(orient='records')
        
        # Clean None values and convert to JSON-serializable format
        cleaned_records = []
        for record in records:
            cleaned = {k: v for k, v in record.items() if pd.notna(v) and v != ""}
            # Convert numpy types to Python types
            for key, value in cleaned.items():
                if isinstance(value, (pd.Timestamp,)):
                    cleaned[key] = value.isoformat()
                elif hasattr(value, 'item'):  # numpy scalar
                    cleaned[key] = value.item()
            cleaned_records.append(cleaned)
        
        logger.info(f"Processed {len(cleaned_records)} records")
        return cleaned_records
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        logger.info("Creating sample data instead")
        return create_sample_cars_data()


def load_faq_csv(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load FAQ data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of FAQ records
    """
    logger.info(f"Loading FAQ data from {file_path}")
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Creating sample FAQ data.")
        return create_sample_faq_data()
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} FAQ entries")
        
        # Normalize column names
        df.columns = [normalize_column_name(col) for col in df.columns]
        
        records = df.to_dict(orient='records')
        cleaned_records = []
        for record in records:
            cleaned = {k: v for k, v in record.items() if pd.notna(v) and v != ""}
            cleaned_records.append(cleaned)
        
        return cleaned_records
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        logger.info("Creating sample FAQ data instead")
        return create_sample_faq_data()


def create_sample_cars_data() -> List[Dict[str, Any]]:
    """Create sample car data if Excel file is not available."""
    logger.info("Creating sample car data")
    return [
        {
            "make": "Toyota",
            "model": "Camry Hybrid",
            "year": 2023,
            "price": 28000,
            "body_type": "Sedan",
            "fuel_type": "Hybrid",
            "mpg_city": 51,
            "mpg_highway": 53,
            "engine_size": 2.5,
            "horsepower": 208,
            "description": "Reliable midsize sedan with excellent fuel economy"
        },
        {
            "make": "Honda",
            "model": "Accord",
            "year": 2023,
            "price": 27000,
            "body_type": "Sedan",
            "fuel_type": "Gasoline",
            "mpg_city": 30,
            "mpg_highway": 38,
            "engine_size": 1.5,
            "horsepower": 192,
            "description": "Comfortable and efficient family sedan"
        },
        {
            "make": "Tesla",
            "model": "Model 3",
            "year": 2023,
            "price": 40000,
            "body_type": "Sedan",
            "fuel_type": "Electric",
            "mpg_city": 132,
            "mpg_highway": 126,
            "engine_size": 0,
            "horsepower": 283,
            "description": "Premium electric sedan with advanced technology"
        },
        {
            "make": "Ford",
            "model": "F-150",
            "year": 2023,
            "price": 35000,
            "body_type": "Truck",
            "fuel_type": "Gasoline",
            "mpg_city": 20,
            "mpg_highway": 24,
            "engine_size": 3.5,
            "horsepower": 400,
            "description": "Powerful full-size pickup truck"
        },
        {
            "make": "BMW",
            "model": "3 Series",
            "year": 2023,
            "price": 45000,
            "body_type": "Sedan",
            "fuel_type": "Gasoline",
            "mpg_city": 26,
            "mpg_highway": 36,
            "engine_size": 2.0,
            "horsepower": 255,
            "description": "Luxury sport sedan with premium features"
        }
    ]


def create_sample_faq_data() -> List[Dict[str, Any]]:
    """Create sample FAQ data if CSV file is not available."""
    logger.info("Creating sample FAQ data")
    return [
        {
            "question": "What is the best fuel-efficient car?",
            "answer": "Hybrid and electric vehicles offer the best fuel efficiency. Popular options include Toyota Camry Hybrid, Honda Accord Hybrid, and Tesla Model 3."
        },
        {
            "question": "How do I choose between sedan and SUV?",
            "answer": "Sedans offer better fuel economy and lower cost, while SUVs provide more space and better off-road capability. Consider your space needs and driving habits."
        },
        {
            "question": "What is the difference between hybrid and electric?",
            "answer": "Hybrid vehicles combine a gasoline engine with an electric motor, while fully electric vehicles run solely on electricity. Electric vehicles have zero emissions but require charging infrastructure."
        }
    ]


def save_processed_data(records: List[Dict[str, Any]], filename: str = "cars_processed.json"):
    """Save processed records to JSON file."""
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(records)} records to {output_path}")


def preview_records(records: List[Dict[str, Any]], num: int = 3):
    """Print preview of records."""
    print(f"\n=== Preview of {min(num, len(records))} records ===")
    for i, record in enumerate(records[:num], 1):
        print(f"\nRecord {i}:")
        for key, value in list(record.items())[:10]:  # Show first 10 fields
            print(f"  {key}: {value}")
        if len(record) > 10:
            print(f"  ... and {len(record) - 10} more fields")


def main():
    """Main function to load and process data."""
    cars_file = DATA_DIR / "cars.xlsx"
    faq_file = DATA_DIR / "faq.csv"
    
    # Load cars data
    cars_data = load_cars_excel(cars_file)
    save_processed_data(cars_data, "cars_processed.json")
    preview_records(cars_data)
    
    # Load FAQ data
    faq_data = load_faq_csv(faq_file)
    save_processed_data(faq_data, "faq_processed.json")
    preview_records(faq_data)
    
    logger.info("Data loading complete!")
    return cars_data, faq_data


if __name__ == "__main__":
    main()

