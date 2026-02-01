#!/usr/bin/env python3
"""
SynthWorld - Open-World Simulation Sandbox

A procedurally generated cyberpunk simulation for robotics research
and synthetic data generation.

Usage:
    python -m synthworld [options]
    
Options:
    --config PATH    Path to configuration file (default: config/settings.yaml)
    --headless       Run without display (for data generation)
    --capture        Enable data capture from start
    --output PATH    Output directory for synthetic data
    --seed INT       Random seed for world generation
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from synthworld.app import SynthWorldApp


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('pybullet').setLevel(logging.WARNING)
    logging.getLogger('panda3d').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SynthWorld - Open-World Simulation Sandbox',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display (for data generation)'
    )
    
    parser.add_argument(
        '--capture',
        action='store_true',
        help='Enable data capture from start'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./synthetic_data',
        help='Output directory for synthetic data'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for world generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--demo',
        type=str,
        choices=['basic', 'robot', 'traffic', 'data'],
        default=None,
        help='Run a specific demo'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SynthWorld - Open-World Simulation Sandbox")
    logger.info("=" * 60)
    
    # Create and run application
    try:
        app = SynthWorldApp(
            config_path=args.config,
            headless=args.headless,
            seed=args.seed
        )
        
        # Enable capture if requested
        if args.capture:
            app.enable_capture(args.output)
        
        # Run demo if specified
        if args.demo:
            app.run_demo(args.demo)
        else:
            app.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error running application: {e}")
        sys.exit(1)
    
    logger.info("SynthWorld shutting down")


if __name__ == '__main__':
    main()
