"""
Smart File Organizer

Automatically organizes files into categorized folders with duplicate detection,
date-based organization, and comprehensive logging.

Author: Gert Coetser
Date: November 2025
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import argparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_organizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FileOrganizer:
    """Organizes files by type, date, and custom rules."""
    
    # Default file type categories
    DEFAULT_CATEGORIES = {
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', 
                      '.xlsx', '.xls', '.pptx', '.ppt', '.csv'],
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', 
                   '.ico', '.tiff', '.webp'],
        'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
                   '.webm', '.m4v'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
        'Code': ['.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', 
                 '.html', '.css', '.php', '.rb', '.go', '.rs'],
        'Executables': ['.exe', '.msi', '.app', '.deb', '.rpm'],
        'Others': []  # Catch-all category
    }
    
    def __init__(self, source_dir: str, config_path: str = None, dry_run: bool = False):
        """
        Initialize the FileOrganizer.
        
        Args:
            source_dir: Directory to organize
            config_path: Path to configuration file
            dry_run: If True, only simulate operations
        """
        self.source_dir = Path(source_dir)
        self.dry_run = dry_run
        self.categories = self.DEFAULT_CATEGORIES.copy()
        self.stats = defaultdict(int)
        
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        
        logger.info(f"Initialized FileOrganizer for: {self.source_dir}")
        if dry_run:
            logger.info("Running in DRY RUN mode - no changes will be made")
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'categories' in config:
                    self.categories.update(config['categories'])
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
    
    def _get_file_category(self, file_path: Path) -> str:
        """
        Determine the category for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Category name
        """
        extension = file_path.suffix.lower()
        
        for category, extensions in self.categories.items():
            if extension in extensions:
                return category
        
        return 'Others'
    
    def _create_category_folder(self, category: str) -> Path:
        """
        Create a folder for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Path to category folder
        """
        folder_path = self.source_dir / f"_Organized_{category}"
        
        if not self.dry_run:
            folder_path.mkdir(exist_ok=True)
            logger.debug(f"Created folder: {folder_path}")
        
        return folder_path
    
    def _get_unique_filename(self, destination: Path) -> Path:
        """
        Generate a unique filename if a file already exists.
        
        Args:
            destination: Target file path
            
        Returns:
            Unique file path
        """
        if not destination.exists():
            return destination
        
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        
        while True:
            new_path = destination.parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def find_duplicates(self) -> Dict[str, List[Path]]:
        """
        Find duplicate files in the source directory.
        
        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        logger.info("Scanning for duplicate files...")
        
        hash_map = defaultdict(list)
        files = [f for f in self.source_dir.rglob('*') if f.is_file()]
        
        for file_path in files:
            file_hash = self._calculate_file_hash(file_path)
            if file_hash:
                hash_map[file_hash].append(file_path)
        
        # Filter to only duplicates
        duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        
        return duplicates
    
    def organize_by_type(self) -> None:
        """Organize files into folders by their type."""
        logger.info("Starting organization by file type...")
        
        files = [f for f in self.source_dir.iterdir() if f.is_file()]
        
        for file_path in files:
            try:
                # Skip hidden files and log files
                if file_path.name.startswith('.') or file_path.name.endswith('.log'):
                    continue
                
                category = self._get_file_category(file_path)
                category_folder = self._create_category_folder(category)
                destination = category_folder / file_path.name
                
                # Handle naming conflicts
                destination = self._get_unique_filename(destination)
                
                # Move file
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would move: {file_path.name} -> {category}/")
                else:
                    shutil.move(str(file_path), str(destination))
                    logger.info(f"Moved: {file_path.name} -> {category}/")
                
                self.stats[category] += 1
                self.stats['total_moved'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                self.stats['errors'] += 1
    
    def organize_by_date(self) -> None:
        """Organize files into folders by their modification date."""
        logger.info("Starting organization by date...")
        
        files = [f for f in self.source_dir.iterdir() if f.is_file()]
        
        for file_path in files:
            try:
                # Skip hidden files and log files
                if file_path.name.startswith('.') or file_path.name.endswith('.log'):
                    continue
                
                # Get file modification date
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                date_folder = mod_time.strftime('%Y-%m')
                
                # Create date folder
                destination_folder = self.source_dir / f"_By_Date_{date_folder}"
                
                if not self.dry_run:
                    destination_folder.mkdir(exist_ok=True)
                
                destination = destination_folder / file_path.name
                destination = self._get_unique_filename(destination)
                
                # Move file
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would move: {file_path.name} -> {date_folder}/")
                else:
                    shutil.move(str(file_path), str(destination))
                    logger.info(f"Moved: {file_path.name} -> {date_folder}/")
                
                self.stats[date_folder] += 1
                self.stats['total_moved'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                self.stats['errors'] += 1
    
    def print_statistics(self) -> None:
        """Print organization statistics."""
        print("\n" + "="*60)
        print("FILE ORGANIZATION SUMMARY")
        print("="*60)
        
        if self.dry_run:
            print("MODE: DRY RUN (No changes made)")
        else:
            print("MODE: Live Organization")
        
        print(f"\nTotal files processed: {self.stats['total_moved']}")
        
        if self.stats['errors'] > 0:
            print(f"Errors encountered: {self.stats['errors']}")
        
        print("\nFiles by category:")
        for category, count in sorted(self.stats.items()):
            if category not in ['total_moved', 'errors']:
                print(f"  {category:20s}: {count:3d} files")
        
        print("="*60 + "\n")
    
    def print_duplicates(self, duplicates: Dict[str, List[Path]]) -> None:
        """
        Print information about duplicate files.
        
        Args:
            duplicates: Dictionary of duplicate files
        """
        if not duplicates:
            print("\n✓ No duplicate files found!")
            return
        
        total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
        total_size = 0
        
        print("\n" + "="*60)
        print(f"DUPLICATE FILES FOUND: {total_duplicates} duplicates")
        print("="*60 + "\n")
        
        for hash_value, paths in duplicates.items():
            file_size = paths[0].stat().st_size / 1024  # KB
            total_size += file_size * (len(paths) - 1)
            
            print(f"Duplicate set ({len(paths)} files, {file_size:.1f} KB each):")
            for path in paths:
                print(f"  - {path}")
            print()
        
        print(f"Potential space savings: {total_size:.1f} KB")
        print("="*60 + "\n")


def create_sample_config() -> None:
    """Create a sample configuration file."""
    sample_config = {
        "categories": {
            "Documents": [".pdf", ".doc", ".docx", ".txt"],
            "Images": [".jpg", ".png", ".gif"],
            "Videos": [".mp4", ".avi", ".mkv"],
            "Audio": [".mp3", ".wav", ".flac"],
            "Archives": [".zip", ".rar", ".7z"],
            "Code": [".py", ".js", ".html", ".css"]
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✓ Created sample configuration file: {config_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Smart File Organizer')
    parser.add_argument('--source', type=str, help='Source directory to organize')
    parser.add_argument('--by-date', action='store_true', help='Organize by date instead of type')
    parser.add_argument('--find-duplicates', action='store_true', help='Find duplicate files')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without making them')
    parser.add_argument('--create-config', action='store_true', help='Create sample config file')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    # Validate source directory
    if not args.source:
        print("Error: --source directory is required")
        print("\nUsage examples:")
        print('  python file_organizer.py --source "C:\\Users\\YourName\\Downloads"')
        print('  python file_organizer.py --source "C:\\path\\to\\folder" --dry-run')
        print('  python file_organizer.py --source "C:\\path\\to\\folder" --by-date')
        print('  python file_organizer.py --source "C:\\path\\to\\folder" --find-duplicates')
        return
    
    try:
        organizer = FileOrganizer(args.source, dry_run=args.dry_run)
        
        # Find duplicates
        if args.find_duplicates:
            duplicates = organizer.find_duplicates()
            organizer.print_duplicates(duplicates)
            return
        
        # Organize files
        if args.by_date:
            organizer.organize_by_date()
        else:
            organizer.organize_by_type()
        
        # Print statistics
        organizer.print_statistics()
        
        if args.dry_run:
            print("This was a dry run. Run without --dry-run to make actual changes.")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
