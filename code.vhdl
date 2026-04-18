library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_layer is
    generic (
        -- Image and Kernel Parameters
        IMG_WIDTH  : positive := 64;  -- Width of the input image
        IMG_HEIGHT : positive := 64;  -- Height of the input image
        KERNEL_SIZE : positive := 3;   -- Size of the square convolution kernel (e.g., 3x3)
        NUM_FILTERS : positive := 8;   -- Number of convolutional filters
        -- Data Widths
        DATA_WIDTH : positive := 8;   -- Data width for pixel values and weights
        ACC_WIDTH  : positive := 16   -- Data width for the accumulator
    );
    port (
        -- Clock and Reset
        clk    : in  std_logic;
        reset  : in  std_logic;
        -- Input Image Data
        img_in : in  std_logic_vector((IMG_WIDTH * IMG_HEIGHT * DATA_WIDTH) - 1 downto 0);
        -- Kernel Weights (Stored in a 1D array)
        kernel_weights : in  std_logic_vector((NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE * DATA_WIDTH) - 1 downto 0);
        -- Output Feature Map
        feature_map_out : out std_logic_vector(((IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_HEIGHT - KERNEL_SIZE + 1) * NUM_FILTERS * DATA_WIDTH) - 1 downto 0);
        -- Ready signal to indicate completion
        ready: out std_logic
    );
end conv_layer;

architecture rtl of conv_layer is

    -- 1. Intermediate Signals and Types ------------------------------------
    type image_buffer_type is array (0 to IMG_HEIGHT - 1, 0 to IMG_WIDTH - 1) of signed(DATA_WIDTH - 1 downto 0);
    type kernel_buffer_type is array (0 to NUM_FILTERS - 1, 0 to KERNEL_SIZE - 1, 0 to KERNEL_SIZE - 1) of signed(DATA_WIDTH - 1 downto 0);
    type feature_map_buffer_type is array (0 to (IMG_HEIGHT - KERNEL_SIZE), 0 to (IMG_WIDTH - KERNEL_SIZE), 0 to NUM_FILTERS - 1) of signed(DATA_WIDTH - 1 downto 0);

    signal image_buffer : image_buffer_type;
    signal kernel_buffer : kernel_buffer_type;
    signal feature_map_buffer : feature_map_buffer_type;

    signal current_row, current_col, current_filter : natural := 0;
    signal output_row, output_col, output_filter : natural := 0;
    signal buffer_ready : std_logic := '0';

begin

    -- 2. Load the input image into the image buffer --------------------------
    process (clk)
        variable img_index : integer := 0;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                img_index := 0;
                current_row := 0;
            else
                if img_index < (IMG_WIDTH * IMG_HEIGHT) then
                    image_buffer(current_row, img_index mod IMG_WIDTH) <= signed(img_in((img_index + 1) * DATA_WIDTH - 1 downto img_index * DATA_WIDTH));
                    img_index := img_index + 1;
                    if img_index mod IMG_WIDTH = 0 then
                        current_row := current_row + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- 3. Load the kernel weights into the kernel buffer ----------------------
    process (clk)
        variable kernel_index : integer := 0;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                kernel_index := 0;
            else
                if kernel_index < (NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE) then
                    kernel_buffer(kernel_index / (KERNEL_SIZE * KERNEL_SIZE), (kernel_index mod (KERNEL_SIZE * KERNEL_SIZE)) / KERNEL_SIZE, kernel_index mod KERNEL_SIZE) <= signed(kernel_weights((kernel_index + 1) * DATA_WIDTH - 1 downto kernel_index * DATA_WIDTH));
                    kernel_index := kernel_index + 1;
                end if;
            end if;
        end if;
    end process;

    -- 4. Perform Convolution and Apply ReLU Activation ----------------------
    process (clk)
        variable acc : signed(ACC_WIDTH - 1 downto 0) := (others => '0');
    begin
        if rising_edge(clk) then
            if reset = '1' then
                acc := (others => '0');
                current_row := 0;
                current_col := 0;
                current_filter := 0;
                output_row := 0;
                output_col := 0;
                output_filter := 0;
                buffer_ready <= '0';
            elsif current_row <= IMG_HEIGHT - KERNEL_SIZE then
                if current_col <= IMG_WIDTH - KERNEL_SIZE then
                    if current_filter < NUM_FILTERS then
                        acc := (others => '0');
                        -- Perform convolution: sum of element-wise multiplications
                        for i in 0 to KERNEL_SIZE - 1 loop
                            for j in 0 to KERNEL_SIZE - 1 loop
                                acc := acc + (image_buffer(current_row + i, current_col + j) * kernel_buffer(current_filter, i, j));
                            end loop;
                        end loop;
                        -- Apply ReLU Activation
                        if acc < 0 then
                            feature_map_buffer(output_row, output_col, output_filter) <= (others => '0');
                        else
                            feature_map_buffer(output_row, output_col, output_filter) <= acc(DATA_WIDTH-1 downto 0);
                        end if;

                        -- Update output indices
                        if output_filter = NUM_FILTERS - 1 then
                            output_filter := 0;
                            if output_col = IMG_WIDTH - KERNEL_SIZE then
                                output_col := 0;
                                output_row := output_row + 1;
                            else
                                output_col := output_col + 1;
                            end if;
                        else
                            output_filter := output_filter + 1;
                        end if;

                        -- Update current filter
                        if current_filter = NUM_FILTERS - 1 then
                            current_filter := 0;
                            if current_col = IMG_WIDTH - KERNEL_SIZE then
                                current_col := 0;
                                current_row := current_row + 1;
                            else
                                current_col := current_col + 1;
                            end if;
                        else
                            current_filter := current_filter + 1;
                        end if;
                    end if;
                else
                    current_col := 0;
                    current_row := current_row + 1;
                end if;
            else
                buffer_ready <= '1'; -- convolution finished
            end if;
        end if;
    end process;

    -- 5. Output Feature Map Conversion to std_logic_vector ------------------
    process (clk)
        variable out_index : integer := 0;
        variable row, col, filt : integer;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                out_index := 0;
                ready <= '0';
            elsif buffer_ready = '1' then
                if out_index < ((IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_HEIGHT - KERNEL_SIZE + 1) * NUM_FILTERS) then
                    -- Convert flat index to 3D indices
                    row := out_index / ((IMG_WIDTH - KERNEL_SIZE + 1) * NUM_FILTERS);
                    col := (out_index mod ((IMG_WIDTH - KERNEL_SIZE + 1) * NUM_FILTERS)) / NUM_FILTERS;
                    filt := out_index mod NUM_FILTERS;

                    feature_map_out(out_index * DATA_WIDTH + DATA_WIDTH - 1 downto out_index * DATA_WIDTH) <= std_logic_vector(feature_map_buffer(row, col, filt));
                    out_index := out_index + 1;
                else
                    ready <= '1'; -- output is ready
                end if;
            else
                ready <= '0';
            end if;
        end if;
    end process;

end rtl;
