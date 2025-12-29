#!/usr/bin/env python3
"""
Modified Molecular-MNIST Dataset Analysis Tool

This script analyzes the Molecular-MNIST dataset with specific visualizations:
- Sample images
- Principal Component Analysis (1500 components)
- Class imbalance and extremes analysis

Dataset files:
- molecular_shape.npy: (20000, 40000, 1) - 20000 images of 200x200 pixels each
- diffraction_pattern.npy: (20000, 40000, 1) - corresponding diffraction patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import warnings
import gc

warnings.filterwarnings('ignore')

# Set global font to Times New Roman and increase size
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20  # Double the original size
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 26
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20


class MolecularMNISTAnalyzer:
    def __init__(self, shape_file='molecular_shape.npy', diffraction_file='diffraction_pattern.npy'):
        """
        Initialize the analyzer with dataset files.

        Args:
            shape_file (str): Path to molecular shape dataset
            diffraction_file (str): Path to diffraction pattern dataset
        """
        self.shape_file = shape_file
        self.diffraction_file = diffraction_file
        self.shape_data = None
        self.diffraction_data = None
        self.img_size = 200  # 200x200 images
        self.n_molecules = 10
        self.variants_per_molecule = 2000

    def load_data(self):
        """Load the molecular shape and diffraction pattern datasets with memory optimization."""
        try:
            print("Loading molecular shape data...")
            # Use memory mapping to avoid loading entire array into memory
            shape_data_raw = np.load(self.shape_file, mmap_mode='r')
            print(f"Shape data dimensions: {shape_data_raw.shape}")

            # Handle extra dimension if present
            if len(shape_data_raw.shape) == 3 and shape_data_raw.shape[2] == 1:
                # Squeeze out the singleton dimension
                self.shape_data = np.array(shape_data_raw[:, :, 0], dtype=np.float32)
            elif len(shape_data_raw.shape) == 2:
                self.shape_data = np.array(shape_data_raw, dtype=np.float32)
            else:
                self.shape_data = np.array(shape_data_raw, dtype=np.float32)

            print(f"Shape data loaded and processed: {self.shape_data.shape}")

            # Clear memory
            del shape_data_raw
            gc.collect()

            print("Loading diffraction pattern data...")
            # Use memory mapping for diffraction data too
            diffraction_data_raw = np.load(self.diffraction_file, mmap_mode='r')
            print(f"Diffraction data dimensions: {diffraction_data_raw.shape}")

            # Handle extra dimension if present
            if len(diffraction_data_raw.shape) == 3 and diffraction_data_raw.shape[2] == 1:
                # Squeeze out the singleton dimension
                self.diffraction_data = np.array(diffraction_data_raw[:, :, 0], dtype=np.float32)
            elif len(diffraction_data_raw.shape) == 2:
                self.diffraction_data = np.array(diffraction_data_raw, dtype=np.float32)
            else:
                self.diffraction_data = np.array(diffraction_data_raw, dtype=np.float32)

            print(f"Diffraction data loaded and processed: {self.diffraction_data.shape}")

            # Clear memory
            del diffraction_data_raw
            gc.collect()

            print("Data loading completed successfully!")
            return True

        except MemoryError:
            print("Memory Error: Dataset too large. Trying alternative loading strategy...")
            return self.load_data_memory_efficient()
        except FileNotFoundError as e:
            print(f"Error: Could not find dataset files. {e}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def load_data_memory_efficient(self):
        """Alternative memory-efficient loading strategy using memory mapping."""
        try:
            print("Using memory-mapped arrays for large datasets...")

            # Load as memory-mapped arrays (not loaded into RAM)
            shape_mmap = np.load(self.shape_file, mmap_mode='r')
            diffraction_mmap = np.load(self.diffraction_file, mmap_mode='r')

            # Handle dimensions
            if len(shape_mmap.shape) == 3 and shape_mmap.shape[2] == 1:
                # Create views without loading into memory
                self.shape_data = shape_mmap[:, :, 0]
            else:
                self.shape_data = shape_mmap

            if len(diffraction_mmap.shape) == 3 and diffraction_mmap.shape[2] == 1:
                self.diffraction_data = diffraction_mmap[:, :, 0]
            else:
                self.diffraction_data = diffraction_mmap

            print(f"Shape data (memory-mapped): {self.shape_data.shape}")
            print(f"Diffraction data (memory-mapped): {self.diffraction_data.shape}")

            print("Memory-mapped loading completed successfully!")
            return True

        except Exception as e:
            print(f"Error in memory-efficient loading: {e}")
            return False

    def reshape_to_images(self, data):
        """Reshape flattened data back to 200x200 images."""
        if len(data.shape) == 1:
            # Single flattened image
            return data.reshape(self.img_size, self.img_size)
        else:
            # Multiple flattened images
            return data.reshape(-1, self.img_size, self.img_size)

    def get_molecule_labels(self):
        """Generate labels for each molecule type."""
        labels = []
        molecule_names = [
            '2x2 Diamond', '3x3 Diamond', '4x4 Diamond', '5x5 Diamond',
            '6x6 Diamond', '7x7 Diamond', '8x8 Diamond', '9x9 Diamond',
            '24-chain Hex', '36-chain Hex'
        ]

        for i, name in enumerate(molecule_names):
            labels.extend([i] * self.variants_per_molecule)

        return np.array(labels), molecule_names

    def display_sample_images(self):
        """Display sample images from the dataset."""
        if self.shape_data is None or self.diffraction_data is None:
            print("Please load data first!")
            return

        print("Reshaping images for display...")
        shape_images = self.reshape_to_images(self.shape_data)
        diffraction_images = self.reshape_to_images(self.diffraction_data)
        labels, molecule_names = self.get_molecule_labels()

        # Use proper aspect ratio for 4x5 grid layout
        # 4 rows, 5 columns needs wider figure
        fig_height = 14  # Increased height to accommodate colorbars
        fig_width = fig_height * 5 / 4 * 1.2  # Wider to accommodate colorbars
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Display samples from each molecule type (4x5 grid)
        # First two rows: shape images
        # Last two rows: diffraction images
        for i in range(self.n_molecules):
            # Get first sample of each molecule
            sample_idx = i * self.variants_per_molecule

            # Determine row and column
            row, col = i // 5, i % 5

            # Shape images (rows 0 and 1)
            ax_shape = plt.subplot(4, 5, row * 5 + col + 1)
            im_shape = ax_shape.imshow(shape_images[sample_idx], cmap='viridis', aspect='equal')
            ax_shape.set_title(f'{molecule_names[i]}\n(Shape)', fontsize=19)
            ax_shape.axis('off')

            # Add colorbar for shape images
            divider = make_axes_locatable(ax_shape)
            cax_shape = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im_shape, cax=cax_shape)

            # Diffraction images (rows 2 and 3)
            ax_diff = plt.subplot(4, 5, (row + 2) * 5 + col + 1)
            im_diff = ax_diff.imshow(diffraction_images[sample_idx], cmap='hot', aspect='equal')
            ax_diff.set_title(f'{molecule_names[i]}\n(Diffraction)', fontsize=19)
            ax_diff.axis('off')

            # Add colorbar for diffraction images
            divider = make_axes_locatable(ax_diff)
            cax_diff = divider.append_axes("right", size="5%", pad=0.1)

            # ç‰¹æ®Šå¤„ç†ç¬¬3è¡Œç¬¬1å¼ diffractionå­å›¾ï¼ˆå³2x2 Diamondçš„Diffractionï¼‰
            if i == 0:  # è¿™æ˜¯ç¬¬3è¡Œç¬¬1å¼ å­å›¾ï¼ˆ2x2 Diamondï¼‰
                cbar = plt.colorbar(im_diff, cax=cax_diff)
                # ä½¿ç"¨ç§'å­¦è®¡æ•°æ³•æ ¼å¼åŒ–è‰²æ¡æ ‡ç­¾
                cbar.formatter.set_powerlimits((0, 0))
                cbar.update_ticks()
            else:
                plt.colorbar(im_diff, cax=cax_diff)

        plt.tight_layout()
        # ä¿å­˜ä¸ºä¸‰ç§æ ¼å¼
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.savefig('sample_images.svg', format='svg', bbox_inches='tight')
        plt.savefig('sample_images.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def analyze_class_imbalance_and_extremes(self):
        """Analyze class imbalance and data extremes - showing 4 key subplots."""
        if self.shape_data is None or self.diffraction_data is None:
            print("Please load data first!")
            return

        print("Creating imbalance_extremes figure...")
        labels, molecule_names = self.get_molecule_labels()

        # Increase height and adjust aspect ratio for better spacing
        fig_height = 12  # Increased height
        fig_width = fig_height * 4 / 3  # 4:3 aspect ratio

        # ä½¿ç"¨gridspec_kwè°ƒæ•´è¡Œé«˜æ¯"ä¾‹ï¼Œä½¿å­å›¾çºµå'æ‹‰é«˜1.2å€
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height),
                                 gridspec_kw={'height_ratios': [1.2, 1.2]})

        datasets = [self.shape_data, self.diffraction_data]
        dataset_names = ['Molecular Shape', 'Diffraction Pattern']

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        subplot_idx = 0

        # Calculate sample means for outlier detection
        sample_means_list = []
        for data, name in zip(datasets, dataset_names):
            print(f"Calculating statistics for {name}...")

            # Calculate means in batches to avoid memory issues
            batch_size = 1000
            sample_means = []

            for batch_start in range(0, data.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, data.shape[0])
                batch_data = np.array(data[batch_start:batch_end])
                batch_means = np.mean(batch_data, axis=1)
                sample_means.extend(batch_means)

            sample_means = np.array(sample_means)
            sample_means_list.append(sample_means)

        print("Creating outlier detection plots...")
        # Subplots 1-2: Outlier detection scatter plots
        for i, (sample_means, name) in enumerate(zip(sample_means_list, dataset_names)):
            # Identify outliers using IQR method
            Q1 = np.percentile(sample_means, 25)
            Q3 = np.percentile(sample_means, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (sample_means < lower_bound) | (sample_means > upper_bound)

            # Plot extreme value distribution
            # For diffraction pattern, scale y-axis by 10000
            if i == 1:  # Diffraction Pattern
                scatter = axes[0, i].scatter(range(len(sample_means)), sample_means / 10000,
                                             c=outliers, cmap='RdYlBu', alpha=0.6, s=10)
                # ä¿®æ"¹è™šçº¿é¢œè‰²å'Œä½ç½®
                axes[0, i].axhline(y=lower_bound / 10000, color='blue', linestyle='--', alpha=0.7, label='Lower bound')
                axes[0, i].axhline(y=upper_bound / 10000, color='green', linestyle='--', alpha=0.7, label='Upper bound')
                axes[0, i].set_ylabel('Mean Intensity (×10$^4$)')
            else:  # Molecular Shape
                scatter = axes[0, i].scatter(range(len(sample_means)), sample_means,
                                             c=outliers, cmap='RdYlBu', alpha=0.6, s=10)
                # ä¿®æ"¹è™šçº¿é¢œè‰²å'Œä½ç½®
                axes[0, i].axhline(y=lower_bound, color='blue', linestyle='--', alpha=0.7, label='Lower bound')
                axes[0, i].axhline(y=upper_bound, color='green', linestyle='--', alpha=0.7, label='Upper bound')
                axes[0, i].set_ylabel('Mean Intensity')

            axes[0, i].set_title(f'{name} Outliers')
            axes[0, i].set_xlabel('Sample Index')
            # MODIFICATION 1: Move legend to lower right corner, slightly upward
            axes[0, i].legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.0, 0.02))
            axes[0, i].grid(True, alpha=0.3)

        print("Creating box plots...")
        # Subplots 3-4: Box plots for each molecule type
        for i, (sample_means, name) in enumerate(zip(sample_means_list, dataset_names)):
            molecule_data = []
            for mol_idx in range(self.n_molecules):
                start_idx = mol_idx * self.variants_per_molecule
                end_idx = (mol_idx + 1) * self.variants_per_molecule
                mol_means = sample_means[start_idx:end_idx]
                if i == 1:  # Diffraction Pattern - scale by 10000
                    molecule_data.append((mol_means / 10000).flatten())
                else:
                    molecule_data.append(mol_means.flatten())

            bp = axes[1, i].boxplot(molecule_data,
                                    labels=[f'M{j + 1}' for j in range(self.n_molecules)],
                                    patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, self.n_molecules))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[1, i].set_title(f'{name} Distribution')
            axes[1, i].set_xlabel('Molecule Type')
            if i == 1:  # Diffraction Pattern
                axes[1, i].set_ylabel('Mean Intensity (×10$^4$)')
            else:
                axes[1, i].set_ylabel('Mean Intensity')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].tick_params(axis='x', rotation=45)

        # Adjust layout first with increased vertical spacing between rows
        plt.tight_layout(rect=[0, 0.08, 1, 0.97], h_pad=4.0)  # Increased h_pad for more space between rows

        # Add subplot labels AFTER tight_layout to ensure proper positioning
        subplot_idx = 0
        for i in range(2):
            for j in range(2):
                if i == 0:  # Top row (a, b) - move up
                    y_pos = -0.25  # Moved up from -0.3
                else:  # Bottom row (c, d) - move down more
                    y_pos = -0.35  # Moved down further from -0.25

                axes[i, j].text(0.5, y_pos, subplot_labels[subplot_idx],
                                transform=axes[i, j].transAxes,
                                ha='center', va='top', fontsize=24, fontweight='bold')
                subplot_idx += 1

        # Force drawing of the figure
        fig.canvas.draw()

        # Save the figure in three formats
        print("Saving imbalance_extremes figures...")
        plt.savefig('imbalance_extremes.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.savefig('imbalance_extremes.svg', format='svg', bbox_inches='tight', pad_inches=0.2)
        plt.savefig('imbalance_extremes.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
        print("Successfully saved imbalance_extremes figures in PNG, SVG and PDF formats")

        # Display the figure
        plt.show()

    def perform_pca_analysis(self):
        """Perform PCA analysis with 1500 components."""
        if self.shape_data is None or self.diffraction_data is None:
            print("Please load data first!")
            return

        print("\nPerforming PCA Analysis (1500 components)...")
        labels, molecule_names = self.get_molecule_labels()

        # Increase height for better spacing and text visibility
        fig_height = 12  # Increased from 10
        fig_width = fig_height * 4 / 3  # 4:3 aspect ratio

        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height),
                                 gridspec_kw={'height_ratios': [1.2, 1.2]})

        datasets = [self.shape_data, self.diffraction_data]
        dataset_names = ['Molecular Shape', 'Diffraction Pattern']

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        subplot_idx = 0

        for i, (data, name) in enumerate(zip(datasets, dataset_names)):
            print(f"Processing {name}: {data.shape}")

            # Use a subset of data for PCA to manage memory
            max_samples = 5000  # Limit for computational efficiency
            if data.shape[0] > max_samples:
                indices = np.random.choice(data.shape[0], max_samples, replace=False)
                # Convert to regular array if memory-mapped
                data_subset = np.array(data[indices], dtype=np.float32)
            else:
                data_subset = np.array(data, dtype=np.float32)

            print(f"Using {data_subset.shape[0]} samples for PCA")

            # Ensure data is 2D (samples x features)
            if len(data_subset.shape) > 2:
                data_subset = data_subset.reshape(data_subset.shape[0], -1)

            # Standardize the data
            print("Standardizing data...")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)

            # Apply PCA with 1500 components
            n_components = min(1500, data_subset.shape[0] - 1, data_subset.shape[1])
            print(f"Applying PCA with {n_components} components...")

            pca = PCA(n_components=n_components)
            data_pca = pca.fit_transform(data_scaled)

            # MODIFICATION 2: Plot explained variance ratio with markers every 2 components
            x_points = range(1, 101, 2)  # Every 2 components from 1 to 100
            y_points = pca.explained_variance_ratio_[::2][:len(x_points)]  # Get corresponding y values
            axes[i, 0].plot(range(1, 101),
                            pca.explained_variance_ratio_[:100],
                            'b-', linewidth=1)
            axes[i, 0].plot(x_points, y_points, 'bo', markersize=4)  # Add markers every 2 components
            axes[i, 0].set_title(f'{name}')
            axes[i, 0].set_xlabel('Component')
            axes[i, 0].set_ylabel('Explained Variance')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xlim([0, 100])
            axes[i, 0].set_ylim(bottom=0)

            # MODIFICATION 2: Plot cumulative explained variance with markers every 30 components
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            x_points_cum = range(1, n_components + 1, 30)  # Every 30 components
            y_points_cum = cumsum[::30][:len(x_points_cum)]  # Get corresponding y values
            axes[i, 1].plot(range(1, n_components + 1),
                            cumsum,
                            'r-', linewidth=1)
            axes[i, 1].plot(x_points_cum, y_points_cum, 'ro', markersize=4)  # Add markers every 30 components
            axes[i, 1].axhline(y=0.95, color='black', linestyle='--', alpha=0.7, label='95%')
            axes[i, 1].set_title(f'{name}')
            axes[i, 1].set_xlabel('Component')
            axes[i, 1].set_ylabel('Cumulative Variance')
            axes[i, 1].legend(fontsize=18)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_xlim([0, n_components])
            axes[i, 1].set_ylim([0, 1.0])

            print(f"\n{name} PCA Results:")
            if n_components >= 10:
                print(f"Variance explained by first 10 components: {cumsum[9]:.4f}")
            if n_components >= 20:
                print(f"Variance explained by first 20 components: {cumsum[19]:.4f}")
            if n_components >= 50:
                print(f"Variance explained by first 50 components: {cumsum[49]:.4f}")
            if n_components >= 100:
                print(f"Variance explained by first 100 components: {cumsum[99]:.4f}")
            if n_components >= 500:
                print(f"Variance explained by first 500 components: {cumsum[499]:.4f}")
            if n_components >= 1000:
                print(f"Variance explained by first 1000 components: {cumsum[999]:.4f}")

            # Find components needed for 95% variance
            components_95 = np.argmax(cumsum >= 0.95) + 1 if np.any(cumsum >= 0.95) else n_components
            print(f"Components needed for 95% variance: {components_95}")
            print(f"Total components computed: {n_components}")

            # Clear memory
            del data_subset, data_scaled, data_pca
            gc.collect()

        # Adjust layout with proper spacing using rect parameter and increased spacing between rows
        plt.tight_layout(rect=[0, 0.08, 1, 0.97], h_pad=4.0)  # Increased h_pad for more space between rows

        # Add subplot labels AFTER tight_layout to ensure proper positioning
        subplot_idx = 0
        for i in range(2):
            for j in range(2):
                # Move all labels down slightly as requested
                axes[i, j].text(0.5, -0.25, subplot_labels[subplot_idx],  # Moved down from -0.18 to -0.25
                                transform=axes[i, j].transAxes,
                                ha='center', va='top', fontsize=24, fontweight='bold')
                subplot_idx += 1

        # Save in three formats
        plt.savefig('principle_component_1500.png', dpi=300, bbox_inches='tight')
        plt.savefig('principle_component_1500.svg', format='svg', bbox_inches='tight')
        plt.savefig('principle_component_1500.pdf', format='pdf', bbox_inches='tight')
        print("Saved principle_component_1500 figures in PNG, SVG and PDF formats")
        plt.show()

    def run_analysis(self):
        """Run the selected analysis functions."""
        print("=" * 60)
        print("MOLECULAR-MNIST DATASET ANALYSIS (Modified)")
        print("=" * 60)

        # Load data
        if not self.load_data():
            return

        print("\n1. Displaying sample images...")
        self.display_sample_images()

        print("\n2. Analyzing class imbalance and extremes...")
        self.analyze_class_imbalance_and_extremes()

        print("\n3. Performing PCA analysis (1500 components)...")
        self.perform_pca_analysis()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED")
        print("All figures saved in PNG, SVG and PDF formats")
        print("=" * 60)


def main():
    """Main function to run the analysis."""
    # Create analyzer instance
    analyzer = MolecularMNISTAnalyzer()

    # Run selected analyses
    analyzer.run_analysis()


if __name__ == "__main__":
    main()