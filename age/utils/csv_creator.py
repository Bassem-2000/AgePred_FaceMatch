import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings

from config.config import config


def create_utkface_csv(dataset_folder: Optional[str] = None, 
                      output_csv_path: Optional[str] = None,
                      max_age: int = None, min_age: int = 0,
                      validate_images: bool = True) -> str:
    """
    Create a CSV file from UTKFace dataset with labels extracted from filenames.
    
    UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    - age: integer from 0 to 116, indicating the age
    - gender: either 0 (male) or 1 (female)
    - race: an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others
    
    Args:
        dataset_folder (str, optional): Path to UTKFace dataset folder
        output_csv_path (str, optional): Path for output CSV file
        max_age (int, optional): Maximum age to include (uses config default)
        min_age (int): Minimum age to include
        validate_images (bool): Whether to validate image files exist
        
    Returns:
        str: Path to created CSV file
    """
    
    # Set default paths
    if dataset_folder is None:
        dataset_folder = config['dataset_root']
    
    if output_csv_path is None:
        csv_dir = config['csv_output_dir']
        os.makedirs(csv_dir, exist_ok=True)
        output_csv_path = os.path.join(csv_dir, config['csv_file_name'])
    
    if max_age is None:
        max_age = config['max_age']
    
    # Validate dataset folder
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(dataset_folder).glob(f"*{ext}"))
        image_files.extend(Path(dataset_folder).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files in {dataset_folder}")
    
    # CSV header
    header = ['image_name', 'age', 'ethnicity', 'gender']
    
    # Ethnicity mapping
    ethnicity_map = {
        0: 'White',
        1: 'Black', 
        2: 'Asian',
        3: 'Indian',
        4: 'Others'
    }
    
    # Gender mapping
    gender_map = {
        0: 'Male',
        1: 'Female'
    }
    
    # Statistics
    processed_count = 0
    skipped_count = 0
    error_count = 0
    skipped_reasons = Counter()
    
    # Process images and create CSV
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for image_file in image_files:
            try:
                # Parse filename
                filename = image_file.name
                name_parts = filename.split('_')
                
                # Validate filename format
                if len(name_parts) < 3:
                    skipped_count += 1
                    skipped_reasons['invalid_filename_format'] += 1
                    continue
                
                # Extract age, gender, ethnicity
                try:
                    age = int(name_parts[0])
                    gender_code = int(name_parts[1])
                    ethnicity_code = int(name_parts[2])
                except (ValueError, IndexError) as e:
                    skipped_count += 1
                    skipped_reasons['invalid_numeric_values'] += 1
                    continue
                
                # Validate ranges
                if age < min_age or age > max_age:
                    skipped_count += 1
                    skipped_reasons['age_out_of_range'] += 1
                    continue
                
                if gender_code not in [0, 1]:
                    skipped_count += 1
                    skipped_reasons['invalid_gender_code'] += 1
                    continue
                
                if ethnicity_code not in [0, 1, 2, 3, 4]:
                    skipped_count += 1
                    skipped_reasons['invalid_ethnicity_code'] += 1
                    continue
                
                # Validate image file exists (if requested)
                if validate_images and not image_file.exists():
                    skipped_count += 1
                    skipped_reasons['missing_image_file'] += 1
                    continue
                
                # Map codes to labels
                gender = gender_map[gender_code]
                ethnicity = ethnicity_map[ethnicity_code]
                
                # Write to CSV
                data = [filename, age, ethnicity, gender]
                writer.writerow(data)
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error processing {image_file.name}: {e}")
                continue
    
    # Print statistics
    print(f"\nCSV Creation Summary:")
    print(f"  Total files found: {len(image_files)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    if skipped_count > 0:
        print(f"\nSkip reasons:")
        for reason, count in skipped_reasons.items():
            print(f"  {reason}: {count}")
    
    print(f"\nCSV file created: {output_csv_path}")
    
    # Validate the created CSV
    try:
        df = pd.read_csv(output_csv_path)
        print(f"CSV validation: {len(df)} rows created successfully")
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Age range: {df['age'].min()} - {df['age'].max()}")
        print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
        print(f"  Ethnicity distribution: {df['ethnicity'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Warning: Could not validate CSV file: {e}")
    
    return output_csv_path


def validate_dataset(csv_path: str, dataset_folder: Optional[str] = None) -> Dict[str, any]:
    """
    Validate dataset integrity by checking CSV and image files.
    
    Args:
        csv_path (str): Path to CSV file
        dataset_folder (str, optional): Path to dataset folder
        
    Returns:
        Dict containing validation results
    """
    
    if dataset_folder is None:
        dataset_folder = config['dataset_root']
    
    print(f"Validating dataset...")
    print(f"  CSV file: {csv_path}")
    print(f"  Dataset folder: {dataset_folder}")
    
    validation_results = {
        'csv_valid': False,
        'missing_images': [],
        'invalid_rows': [],
        'duplicate_images': [],
        'statistics': {}
    }
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"  Loaded CSV with {len(df)} rows")
        
        # Check required columns
        required_columns = ['image_name', 'age', 'ethnicity', 'gender']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  Error: Missing required columns: {missing_columns}")
            return validation_results
        
        validation_results['csv_valid'] = True
        
        # Check for missing images
        missing_images = []
        for idx, row in df.iterrows():
            image_path = os.path.join(dataset_folder, row['image_name'])
            if not os.path.exists(image_path):
                missing_images.append(row['image_name'])
        
        validation_results['missing_images'] = missing_images
        
        # Check for invalid data
        invalid_rows = []
        for idx, row in df.iterrows():
            issues = []
            
            # Age validation
            if not isinstance(row['age'], (int, float)) or row['age'] < 0 or row['age'] > 120:
                issues.append('invalid_age')
            
            # Gender validation
            if row['gender'] not in ['Male', 'Female']:
                issues.append('invalid_gender')
            
            # Ethnicity validation
            valid_ethnicities = ['White', 'Black', 'Asian', 'Indian', 'Others']
            if row['ethnicity'] not in valid_ethnicities:
                issues.append('invalid_ethnicity')
            
            if issues:
                invalid_rows.append({'index': idx, 'image_name': row['image_name'], 'issues': issues})
        
        validation_results['invalid_rows'] = invalid_rows
        
        # Check for duplicate images
        duplicate_images = df[df.duplicated(subset=['image_name'], keep=False)]['image_name'].tolist()
        validation_results['duplicate_images'] = list(set(duplicate_images))
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_samples': len(df),
            'missing_images_count': len(missing_images),
            'invalid_rows_count': len(invalid_rows),
            'duplicate_images_count': len(duplicate_images),
            'age_stats': {
                'min': df['age'].min(),
                'max': df['age'].max(),
                'mean': df['age'].mean(),
                'std': df['age'].std()
            },
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'ethnicity_distribution': df['ethnicity'].value_counts().to_dict()
        }
        
        # Print validation summary
        print(f"\nValidation Results:")
        print(f"  Total samples: {validation_results['statistics']['total_samples']}")
        print(f"  Missing images: {validation_results['statistics']['missing_images_count']}")
        print(f"  Invalid rows: {validation_results['statistics']['invalid_rows_count']}")
        print(f"  Duplicate images: {validation_results['statistics']['duplicate_images_count']}")
        
        if missing_images:
            print(f"\nFirst 10 missing images:")
            for img in missing_images[:10]:
                print(f"    {img}")
        
        if invalid_rows:
            print(f"\nFirst 10 invalid rows:")
            for row in invalid_rows[:10]:
                print(f"    {row['image_name']}: {', '.join(row['issues'])}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        validation_results['error'] = str(e)
    
    return validation_results


def analyze_dataset_statistics(csv_path: str, save_plots: bool = True, 
                             output_dir: Optional[str] = None) -> Dict[str, any]:
    """
    Analyze and visualize dataset statistics.
    
    Args:
        csv_path (str): Path to CSV file
        save_plots (bool): Whether to save plot figures
        output_dir (str, optional): Directory to save plots
        
    Returns:
        Dict containing analysis results
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    print(f"Analyzing dataset statistics from: {csv_path}")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        total_samples = len(df)
        age_stats = df['age'].describe()
        gender_counts = df['gender'].value_counts()
        ethnicity_counts = df['ethnicity'].value_counts()
        
        # Age distribution analysis
        age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        age_groups = pd.cut(df['age'], bins=age_bins, right=False)
        age_group_counts = age_groups.value_counts().sort_index()
        
        # Cross-tabulation analysis
        gender_ethnicity_crosstab = pd.crosstab(df['gender'], df['ethnicity'])
        age_gender_stats = df.groupby('gender')['age'].describe()
        age_ethnicity_stats = df.groupby('ethnicity')['age'].describe()
        
        # Create visualizations
        if save_plots:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Age distribution histogram
            axes[0, 0].hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Age Distribution', fontsize=14)
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gender distribution pie chart
            axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                          colors=['lightcoral', 'lightblue'])
            axes[0, 1].set_title('Gender Distribution', fontsize=14)
            
            # Ethnicity distribution bar chart
            ethnicity_counts.plot(kind='bar', ax=axes[0, 2], color='lightgreen')
            axes[0, 2].set_title('Ethnicity Distribution', fontsize=14)
            axes[0, 2].set_xlabel('Ethnicity')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Age groups distribution
            age_group_counts.plot(kind='bar', ax=axes[1, 0], color='orange')
            axes[1, 0].set_title('Age Groups Distribution', fontsize=14)
            axes[1, 0].set_xlabel('Age Groups')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Age by gender box plot
            df.boxplot(column='age', by='gender', ax=axes[1, 1])
            axes[1, 1].set_title('Age Distribution by Gender', fontsize=14)
            axes[1, 1].set_xlabel('Gender')
            axes[1, 1].set_ylabel('Age')
            
            # Age by ethnicity box plot
            df.boxplot(column='age', by='ethnicity', ax=axes[1, 2])
            axes[1, 2].set_title('Age Distribution by Ethnicity', fontsize=14)
            axes[1, 2].set_xlabel('Ethnicity')
            axes[1, 2].set_ylabel('Age')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = os.path.join(output_dir, 'dataset_analysis.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Analysis plots saved to: {plot_path}")
            
            plt.show()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            
            # Create numerical encoding for categorical variables
            df_encoded = df.copy()
            df_encoded['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
            ethnicity_encoding = {eth: i for i, eth in enumerate(df['ethnicity'].unique())}
            df_encoded['ethnicity_encoded'] = df['ethnicity'].map(ethnicity_encoding)
            
            # Correlation matrix
            corr_matrix = df_encoded[['age', 'gender_encoded', 'ethnicity_encoded']].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=['Age', 'Gender', 'Ethnicity'],
                       yticklabels=['Age', 'Gender', 'Ethnicity'])
            plt.title('Feature Correlation Matrix', fontsize=16)
            
            if save_plots:
                corr_path = os.path.join(output_dir, 'correlation_matrix.png')
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                print(f"Correlation matrix saved to: {corr_path}")
            
            plt.show()
        
        # Compile analysis results
        analysis_results = {
            'total_samples': total_samples,
            'age_statistics': {
                'count': int(age_stats['count']),
                'mean': float(age_stats['mean']),
                'std': float(age_stats['std']),
                'min': int(age_stats['min']),
                'max': int(age_stats['max']),
                'q25': float(age_stats['25%']),
                'median': float(age_stats['50%']),
                'q75': float(age_stats['75%'])
            },
            'gender_distribution': gender_counts.to_dict(),
            'ethnicity_distribution': ethnicity_counts.to_dict(),
            'age_groups': age_group_counts.to_dict(),
            'gender_ethnicity_crosstab': gender_ethnicity_crosstab.to_dict(),
            'age_by_gender': age_gender_stats.to_dict(),
            'age_by_ethnicity': age_ethnicity_stats.to_dict()
        }
        
        # Print detailed statistics
        print(f"\n=== Dataset Analysis Summary ===")
        print(f"Total Samples: {total_samples:,}")
        print(f"\nAge Statistics:")
        print(f"  Range: {int(age_stats['min'])} - {int(age_stats['max'])} years")
        print(f"  Mean: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years")
        print(f"  Median: {age_stats['50%']:.1f} years")
        
        print(f"\nGender Distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nEthnicity Distribution:")
        for ethnicity, count in ethnicity_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {ethnicity}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nAge Statistics by Gender:")
        for gender in age_gender_stats.index:
            stats = age_gender_stats.loc[gender]
            print(f"  {gender}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}, Range=[{int(stats['min'])}-{int(stats['max'])}]")
        
        print(f"\nAge Statistics by Ethnicity:")
        for ethnicity in age_ethnicity_stats.index:
            stats = age_ethnicity_stats.loc[ethnicity]
            print(f"  {ethnicity}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}, Range=[{int(stats['min'])}-{int(stats['max'])}]")
        
        return analysis_results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {'error': str(e)}


def clean_dataset(csv_path: str, output_path: Optional[str] = None,
                 remove_outliers: bool = True, outlier_std_threshold: float = 3.0,
                 balance_gender: bool = False, balance_ethnicity: bool = False) -> str:
    """
    Clean and preprocess the dataset.
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned CSV file
        remove_outliers (bool): Whether to remove age outliers
        outlier_std_threshold (float): Standard deviation threshold for outliers
        balance_gender (bool): Whether to balance gender distribution
        balance_ethnicity (bool): Whether to balance ethnicity distribution
        
    Returns:
        str: Path to cleaned CSV file
    """
    
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_cleaned.csv"
    
    print(f"Cleaning dataset: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"Original dataset size: {original_count}")
    
    # Remove outliers
    if remove_outliers:
        age_mean = df['age'].mean()
        age_std = df['age'].std()
        age_threshold = outlier_std_threshold * age_std
        
        outlier_mask = (np.abs(df['age'] - age_mean) <= age_threshold)
        outliers_removed = len(df) - outlier_mask.sum()
        df = df[outlier_mask]
        
        print(f"Removed {outliers_removed} age outliers (threshold: ±{age_threshold:.1f} years)")
    
    # Balance gender distribution
    if balance_gender:
        gender_counts = df['gender'].value_counts()
        min_gender_count = gender_counts.min()
        
        balanced_dfs = []
        for gender in gender_counts.index:
            gender_df = df[df['gender'] == gender].sample(n=min_gender_count, random_state=42)
            balanced_dfs.append(gender_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"Balanced gender distribution to {min_gender_count} samples per gender")
    
    # Balance ethnicity distribution
    if balance_ethnicity:
        ethnicity_counts = df['ethnicity'].value_counts()
        min_ethnicity_count = ethnicity_counts.min()
        
        balanced_dfs = []
        for ethnicity in ethnicity_counts.index:
            ethnicity_df = df[df['ethnicity'] == ethnicity].sample(n=min_ethnicity_count, random_state=42)
            balanced_dfs.append(ethnicity_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"Balanced ethnicity distribution to {min_ethnicity_count} samples per ethnicity")
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    
    final_count = len(df)
    reduction_percentage = ((original_count - final_count) / original_count) * 100
    
    print(f"Cleaned dataset saved: {output_path}")
    print(f"Final dataset size: {final_count} (reduced by {reduction_percentage:.1f}%)")
    
    return output_path


def export_dataset_info(csv_path: str, output_file: Optional[str] = None) -> str:
    """
    Export comprehensive dataset information to a text file.
    
    Args:
        csv_path (str): Path to CSV file
        output_file (str, optional): Path for output info file
        
    Returns:
        str: Path to exported info file
    """
    
    if output_file is None:
        base_name = os.path.splitext(csv_path)[0]
        output_file = f"{base_name}_info.txt"
    
    # Get analysis results
    analysis = analyze_dataset_statistics(csv_path, save_plots=False)
    validation = validate_dataset(csv_path)
    
    # Write comprehensive info
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("UTKFace Dataset Information Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset CSV: {csv_path}\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Samples: {analysis['total_samples']:,}\n")
        f.write(f"Age Range: {analysis['age_statistics']['min']} - {analysis['age_statistics']['max']} years\n")
        f.write(f"Mean Age: {analysis['age_statistics']['mean']:.1f} ± {analysis['age_statistics']['std']:.1f} years\n")
        f.write(f"Median Age: {analysis['age_statistics']['median']:.1f} years\n\n")
        
        # Gender distribution
        f.write("GENDER DISTRIBUTION\n")
        f.write("-" * 20 + "\n")
        total = analysis['total_samples']
        for gender, count in analysis['gender_distribution'].items():
            percentage = (count / total) * 100
            f.write(f"{gender}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Ethnicity distribution
        f.write("ETHNICITY DISTRIBUTION\n")
        f.write("-" * 25 + "\n")
        for ethnicity, count in analysis['ethnicity_distribution'].items():
            percentage = (count / total) * 100
            f.write(f"{ethnicity}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Validation results
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Missing Images: {validation['statistics']['missing_images_count']}\n")
        f.write(f"Invalid Rows: {validation['statistics']['invalid_rows_count']}\n")
        f.write(f"Duplicate Images: {validation['statistics']['duplicate_images_count']}\n\n")
        
        # Age statistics by demographics
        f.write("AGE STATISTICS BY DEMOGRAPHICS\n")
        f.write("-" * 35 + "\n")
        f.write("By Gender:\n")
        for gender, stats in analysis['age_by_gender'].items():
            f.write(f"  {gender}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}, Range=[{int(stats['min'])}-{int(stats['max'])}]\n")
        
        f.write("\nBy Ethnicity:\n")
        for ethnicity, stats in analysis['age_by_ethnicity'].items():
            f.write(f"  {ethnicity}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}, Range=[{int(stats['min'])}-{int(stats['max'])}]\n")
    
    print(f"Dataset information exported to: {output_file}")
    return output_file


# Main execution function
def main():
    """Main function for running CSV creation and analysis."""
    
    print("UTKFace Dataset CSV Creator and Analyzer")
    print("=" * 50)
    
    try:
        # Create CSV from dataset
        print("\n1. Creating CSV from UTKFace dataset...")
        csv_path = create_utkface_csv()
        
        # Validate dataset
        print("\n2. Validating dataset...")
        validation_results = validate_dataset(csv_path)
        
        # Analyze statistics
        print("\n3. Analyzing dataset statistics...")
        analysis_results = analyze_dataset_statistics(csv_path)
        
        # Export comprehensive info
        print("\n4. Exporting dataset information...")
        info_file = export_dataset_info(csv_path)
        
        print(f"\n✅ Dataset processing completed successfully!")
        print(f"   CSV file: {csv_path}")
        print(f"   Info file: {info_file}")
        
    except Exception as e:
        print(f"❌ Error during dataset processing: {e}")


if __name__ == "__main__":
    main()